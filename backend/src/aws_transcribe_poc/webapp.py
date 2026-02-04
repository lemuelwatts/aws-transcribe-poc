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

"""FastAPI Web App - Meeting Processing API.

This module provides the main API endpoint for processing meeting recordings.
Individual service endpoints are available via routers for testing.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from .models.api_models import (
    MeetingResponse,
    ProcessingMetrics,
    SpeakerMethod,
    TranscriptionResult,
)
from .routers.analysis import router as analysis_router
from .routers.audio import router as audio_router
from .routers.ingestion import router as ingestion_router
from .routers.meeting import router as meeting_router
from .routers.notes import router as notes_router
from .routers.speaker import router as speaker_router
from .routers.transcript import router as transcript_router
from .services.analyzer import AnalyzerService
from .services.input_handler import InputHandler
from .services.meeting_combiner import MeetingCombiner
from .services.notes_normalizer import NotesNormalizer
from .services.s3_handler import S3Handler
from .services.speaker_matcher import SpeakerMatcher
from .services.transcribe import TranscriptionService
from .services.transcript_normalizer import TranscriptNormalizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = FastAPI(
    title="Meeting Processor",
    description="API for processing meeting recordings with transcription, "
    "speaker identification, and analysis",
    version="1.0.0",
)

app.include_router(analysis_router)
app.include_router(audio_router)
app.include_router(ingestion_router)
app.include_router(meeting_router)
app.include_router(notes_router)
app.include_router(transcript_router)
app.include_router(speaker_router)


@app.get("/")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/process_meeting", response_model=MeetingResponse)
async def process_meeting(
    file: UploadFile = File(..., description="Meeting audio/video file"),
    notes_file: UploadFile | None = File(
        None, description="Optional meeting notes text file"
    ),
    speaker_method: SpeakerMethod = Form(
        SpeakerMethod.HYBRID, description="Speaker identification method"
    ),
    similarity_threshold: float = Form(
        0.5, description="Biometric matching threshold (0.0-1.0)"
    ),
    generate_analysis: bool = Form(
        default=True, description="Generate meeting analysis report"
    ),
    save_report: bool = Form(default=True, description="Save analysis report to disk"),
    save_metrics: bool = Form(
        default=False, description="Save processing metrics to S3"
    ),
) -> MeetingResponse:
    """Process a meeting recording.

    This endpoint handles the complete meeting processing pipeline:
    1. Convert audio to WAV and upload to S3
    2. Transcribe with AWS Transcribe
    3. Identify speakers (optional, based on speaker_method)
    4. Generate analysis report (optional, based on generate_analysis)

    Use Cases:
    - Transcription only: speaker_method=none, generate_analysis=false
    - Transcription + speaker ID: speaker_method=llm/biometrics/hybrid
    - Full pipeline (default): Just upload the file
    """
    start_time = time.time()
    tmp_path = None
    s3_uri = ""
    audio_metrics = None

    try:
        # save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        logger.info(f"Processing meeting: {file.filename}")

        # Process audio and upload to S3
        s3_uri, audio_metrics = InputHandler().process_input(tmp_path, file.filename)

        # Transcribe with AWS Transcribe
        results = TranscriptionService().transcribe_all(
            [s3_uri], save_metrics=save_metrics
        )
        if not results or not results[0].success:
            error = results[0].error if results else "No transcription results"
            raise RuntimeError(f"Transcription failed: {error}")

        transcription = results[0]
        raw_transcript = S3Handler().download_json(transcription.s3_output_uri)

        # Normalize transcript and apply speaker names
        normalizer = TranscriptNormalizer()
        normalized = normalizer.normalize(raw_transcript)

        # Identify speakers
        speaker_result = SpeakerMatcher().match(
            method=speaker_method,
            audio_path=tmp_path,
            normalized_transcript=normalized,
            threshold=similarity_threshold,
        )

        if speaker_result.mapping:
            normalized = normalizer.replace_speaker_labels(
                normalized, speaker_result.mapping
            )

        # Combine with notes if provided
        if notes_file:
            notes_text = (await notes_file.read()).decode("utf-8")
            normalized_notes = NotesNormalizer().normalize(notes_text)
            notes_dict = normalized_notes.to_dict()
        else:
            notes_dict = {"attendee_notes": {}}

        combined = MeetingCombiner().combine(normalized.to_dict(), notes_dict)

        # Generate analysis if requested
        analysis_dict = None
        analysis_path = None

        if generate_analysis:
            # Save combined data for analyzer
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            combined_path = output_dir / f"combined_{Path(file.filename).stem}.json"

            with open(combined_path, "w") as f: 
                json.dump(combined.to_dict(), f, indent=2)

            report, analysis_path = AnalyzerService(
                input_file=str(combined_path)
            ).run_analysis(save_report=save_report)

            analysis_dict = report.model_dump()

        duration = time.time() - start_time
        logger.info(f"Completed processing {file.filename} in {duration:.2f}s")

        return MeetingResponse(
            success=True,
            filename=file.filename,
            s3_uri=s3_uri,
            transcription=TranscriptionResult(
                s3_uri=transcription.s3_uri,
                success=transcription.success,
                s3_output_uri=transcription.s3_output_uri,
                s3_summary_uri=transcription.s3_summary_uri,
                transcription_time_seconds=transcription.transcription_duration_seconds,
            ),
            processing_metrics=ProcessingMetrics(
                total_time_seconds=audio_metrics["total_processing_time_seconds"],
                ffmpeg_time_seconds=audio_metrics["ffmpeg_conversion_time_seconds"],
                s3_upload_time_seconds=audio_metrics["s3_upload_time_seconds"],
                s3_upload_speed_mbps=audio_metrics["s3_upload_speed_mbps"],
                input_size_mb=audio_metrics["input_file_size_mb"],
                output_size_mb=audio_metrics["output_file_size_mb"],
                size_reduction_percent=audio_metrics["size_reduction_percent"],
            ),
            speaker_mapping=speaker_result.mapping,
            speaker_details=dict(speaker_result.details),
            unmatched_speakers=speaker_result.unmatched,
            speaker_method_used=speaker_method.value,
            analysis=analysis_dict,
            analysis_path=analysis_path,
            total_duration_seconds=duration,
        )

    except Exception as e:
        logger.error(f"Error processing meeting: {e}")
        duration = time.time() - start_time

        return MeetingResponse(
            success=False,
            filename=file.filename,
            s3_uri=s3_uri,
            transcription=TranscriptionResult(
                s3_uri=s3_uri, success=False, error=str(e)
            ),
            processing_metrics=None,
            total_duration_seconds=duration,
            error=str(e),
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def start_app() -> None:
    """Start the FastAPI webapp."""
    uvicorn.run(app, host="0.0.0.0", port=8888)  # noqa: S104
