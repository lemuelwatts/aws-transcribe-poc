# Copyright 2025 Booz Allen Hamilton.
#
# Booz Allen Hamilton Confidential Information.
#
# The contents of this file are the intellectual property of
# Booz Allen Hamilton, Inc. ("BAH") and are subject to copyright protection
# under the laws of the United States and other countries.
#
# You acknowledge that misappropriation, misuse, or redistribution of content
# on the file could cause irreparable harm to BAH and/or to third parties.
#
# You may not copy, reproduce, distribute, publish, display, execute, modify,
# create derivative works of, transmit, sell or offer for resale, or in any way
# exploit any part of this code or program without BAH's express written permission.
#
# The contents of this code or program contains code
# that is itself or was created using artificial intelligence.
#
# To the best of our knowledge, this code does not infringe third-party intellectual
# property rights, contain errors, inaccuracies, bias, or security concerns.
#
# However, Booz Allen does not warrant, claim, or provide any implied
# or express warranty for the aforementioned, nor of merchantability
# or fitness for purpose.
#
# Booz Allen expressly limits liability, whether by contract, tort or in equity
# for any damage or harm caused by use of this artificial intelligence code or program.
#
# Booz Allen is providing this code or program "as is" with the understanding
# that any separately negotiated standards of performance for said code
# or program will be met for the duration of any applicable contract under which
# the code or program is provided.

"""FastAPI Web App.

This module configures the FastAPI Web Server that provides HTTP/API access
to the rest of the "backend".
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .routers.audio import router as audio_router
from .routers.transcript import router as transcript_router
from .services.analyzer import AnalyzerService
from .services.input_handler import InputHandler
from .services.transcribe import TranscriptionService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


app = FastAPI(
    title="AWS Transcribe POC",
    description="API for transcribing media files using AWS Transcribe",
    version="0.0.1",
)

app.include_router(audio_router)
app.include_router(transcript_router)


class TranscriptionRequestModel(BaseModel):
    """Request model for batch transcription."""

    s3_uris: list[str]
    save_metrics: bool = False


class TranscriptionResultModel(BaseModel):
    """Result model for a single file transcription."""

    s3_uri: str
    success: bool
    s3_output_uri: str | None = None
    s3_summary_uri: str | None = None
    s3_metrics_uri: str | None = None
    transcription_duration_seconds: float | None = None
    summary_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    error: str | None = None


class TranscriptionResponseModel(BaseModel):
    """Response model for batch transcription."""

    total_files: int
    successful: int
    failed: int
    results: list[TranscriptionResultModel]


class PerformanceMetrics(BaseModel):
    """Performance metrics for audio processing."""

    total_processing_time_seconds: float
    input_metadata: dict
    ffmpeg_conversion_time_seconds: float
    s3_upload_time_seconds: float
    s3_upload_speed_mbps: float
    input_file_size_mb: float
    output_file_size_mb: float
    size_reduction_percent: float


class ResponseModel(BaseModel):
    """Response model for end-to-end pipeline."""

    original_filename: str
    s3_uri: str
    processing_metrics: PerformanceMetrics
    transcription_result: TranscriptionResultModel
    total_pipeline_duration_seconds: float


class BatchResponseModel(BaseModel):
    """Response model for batch end-to-end pipeline."""

    total_files: int
    successful: int
    failed: int
    total_pipeline_duration: float
    results: list[ResponseModel]  # collect processing metrics for each file


class AnalyzeRequestModel(BaseModel):
    """Request model for meeting analysis"""

    input_file_path: str
    save_report: bool = False


class AnalyzeResponseModel(BaseModel):
    """Response model for meeting analysis"""

    success: bool
    final_report: dict | None = None
    output_file_path: str | None = None
    error: str | None = None


@app.get("/")
async def root():
    """Root endpoint."""
    return "Healthy"


@app.post("/transcribe", response_model=TranscriptionResponseModel)
async def transcribe_files(
    request: TranscriptionRequestModel,
) -> TranscriptionResponseModel:
    """Transcribe media files from S3.

    Accepts a list of S3 URIs, transcribes them using AWS Transcribe,
    and saves the transcripts to the output/ folder in the same S3 bucket.

    Args:
        request: Request containing list of S3 URIs to transcribe.
            - s3_uris: List of S3 URIs to transcribe.
            - save_metrics: Whether to save metrics JSON to S3 (default: false).

    Requires environment variables:
    - AWS_REGION: AWS region (default: us-east-1)

    Returns:
        TranscriptionResponseModel with summary and individual file results.
    """
    transcription_service = TranscriptionService()
    results = transcription_service.transcribe_all(
        request.s3_uris, save_metrics=request.save_metrics
    )

    result_models = [
        TranscriptionResultModel(
            s3_uri=r.s3_uri,
            success=r.success,
            s3_output_uri=r.s3_output_uri,
            s3_summary_uri=r.s3_summary_uri,
            s3_metrics_uri=r.s3_metrics_uri,
            transcription_duration_seconds=r.transcription_duration_seconds,
            summary_duration_seconds=r.summary_duration_seconds,
            total_duration_seconds=r.total_duration_seconds,
            error=r.error,
        )
        for r in results
    ]

    successful = sum(1 for r in results if r.success)

    return TranscriptionResponseModel(
        total_files=len(results),
        successful=successful,
        failed=len(results) - successful,
        results=result_models,
    )


@app.post("/upload_and_transcribe", response_model=ResponseModel)
async def upload_and_transcribe(
    file: UploadFile = File(...), save_metrics: bool = False
) -> ResponseModel:
    """End to end pipeline: upload, process, convert to WAV, upload to s3, and transcribe."""
    pipeline_start = time.time()

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        handler = InputHandler()
        s3_uri, audio_processing_metrics = handler.process_input(
            tmp_path, file.filename
        )

        transcription_service = TranscriptionService()
        results = transcription_service.transcribe_all(
            [s3_uri], save_metrics=save_metrics
        )

        if not results:
            raise RuntimeError("Transcription service returned no results")

        transcription_result = results[0]

        transcription_result_model = TranscriptionResultModel(
            s3_uri=transcription_result.s3_uri,
            success=transcription_result.success,
            s3_output_uri=transcription_result.s3_output_uri,
            s3_summary_uri=transcription_result.s3_summary_uri,
            s3_metrics_uri=transcription_result.s3_metrics_uri,
            transcription_duration_seconds=transcription_result.transcription_duration_seconds,
            summary_duration_seconds=transcription_result.summary_duration_seconds,
            total_duration_seconds=transcription_result.total_duration_seconds,
            error=transcription_result.error,
        )

        total_duration = time.time() - pipeline_start

        return ResponseModel(
            original_filename=file.filename,
            s3_uri=s3_uri,
            processing_metrics=PerformanceMetrics(**audio_processing_metrics),
            transcription_result=transcription_result_model,
            total_pipeline_duration_seconds=total_duration,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/upload_and_transcribe_batch", response_model=BatchResponseModel)
async def upload_and_transcribe_batch(
    files: list[UploadFile] = File(...), save_metrics: bool = False
) -> BatchResponseModel:
    """End to end pipeline for multiple files: upload, process, convert to WAV, upload to s3, and transcribe."""
    pipeline_start = time.time()
    tmp_paths = []
    file_results = []

    try:
        handler = InputHandler()

        s3_uris = []
        audio_processing_metrics = []

        for file in files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename).suffix
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_paths.append(tmp.name)

                s3_uri, audio_metrics = handler.process_input(tmp.name, file.filename)
                s3_uris.append(s3_uri)

                audio_processing_metrics.append(
                    {
                        "filename": file.filename,
                        "s3_uri": s3_uri,
                        "metrics": audio_metrics,
                    }
                )

        transcription_service = TranscriptionService()
        results = transcription_service.transcribe_all(
            s3_uris, save_metrics=save_metrics
        )

        for metric, result in zip(audio_processing_metrics, results):
            transcription_result_model = TranscriptionResultModel(
                s3_uri=result.s3_uri,
                success=result.success,
                s3_output_uri=result.s3_output_uri,
                s3_summary_uri=result.s3_summary_uri,
                s3_metrics_uri=result.s3_metrics_uri,
                transcription_duration_seconds=result.transcription_duration_seconds,
                summary_duration_seconds=result.summary_duration_seconds,
                total_duration_seconds=result.total_duration_seconds,
                error=result.error,
            )

            file_results.append(
                ResponseModel(
                    original_filename=metric["filename"],
                    s3_uri=metric["s3_uri"],
                    processing_metrics=PerformanceMetrics(**metric["metrics"]),
                    transcription_result=transcription_result_model,
                    total_pipeline_duration_seconds=0,
                )
            )

        successful = sum(1 for r in file_results if r.transcription_result.success)
        total_duration = time.time() - pipeline_start

        return BatchResponseModel(
            total_files=len(file_results),
            successful=successful,
            failed=len(file_results) - successful,
            total_pipeline_duration=total_duration,
            results=file_results,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@app.post("/analyze", response_model=AnalyzeResponseModel)  
async def analyze_meeting(request: AnalyzeRequestModel) -> AnalyzeResponseModel:
    """Analyze a meeting transcript and generate insights.

    Takes in a JSON file containing transcript and user notes and generates
    a comprehensive analysis including:
    - Summary
    - Action Items
    - Inconsistencies (optional)
    - Compliance issues (optional)
    - Meeting improvements (optional)

    Args:
        request: Request containing:
            - input_file_path: Path to JSON file with meeting context
            - save_report: whether to save report to output/ folder (default: False)

    Returns:
        AnalyzeResponseModel with analysis results and optional file path (if save_report=True)
    """
    try:
        input_path = Path(request.input_file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {request.input_file_path}")

        analyzer_service = AnalyzerService(request.input_file_path)

        report, output_path = analyzer_service.run_analysis(
            save_report=request.save_report
        )

        return AnalyzeResponseModel(
            success=True, report=report.model_dump(), output_file_path=output_path
        )
    except FileNotFoundError as e:
        return AnalyzeResponseModel(success=False, error=str(e))
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return AnalyzeResponseModel(success=False, error=str(e))


def start_app() -> None:
    """Start the FastAPI webapp."""
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
