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

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .services.transcribe import TranscriptionService

app = FastAPI(
    title="AWS Transcribe POC",
    description="API for transcribing media files using AWS Transcribe",
    version="0.0.1",
)


class MyResponseModel(BaseModel):
    """Response model for greeting."""

    content: str = "Hello World"


class TranscriptionRequestModel(BaseModel):
    """Request model for batch transcription."""

    s3_uris: list[str]


class TranscriptionResultModel(BaseModel):
    """Result model for a single file transcription."""

    s3_uri: str
    success: bool
    s3_output_uri: str | None = None
    error: str | None = None


class TranscriptionResponseModel(BaseModel):
    """Response model for batch transcription."""

    total_files: int
    successful: int
    failed: int
    results: list[TranscriptionResultModel]


@app.get("/", response_model=MyResponseModel)
async def root() -> MyResponseModel:
    """Root endpoint."""
    return MyResponseModel(content="Hello World")


@app.post("/transcribe", response_model=TranscriptionResponseModel)
async def transcribe_files(
    request: TranscriptionRequestModel,
) -> TranscriptionResponseModel:
    """Transcribe media files from S3.

    Accepts a list of S3 URIs, transcribes them using AWS Transcribe,
    and saves the transcripts to the output/ folder in the same S3 bucket.

    Args:
        request: Request containing list of S3 URIs to transcribe.

    Requires environment variables:
    - AWS_REGION: AWS region (default: us-east-1)

    Returns:
        TranscriptionResponseModel with summary and individual file results.
    """
    transcription_service = TranscriptionService()
    results = transcription_service.transcribe_all(request.s3_uris)

    result_models = [
        TranscriptionResultModel(
            s3_uri=r.s3_uri,
            success=r.success,
            s3_output_uri=r.s3_output_uri,
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


def start_app() -> None:
    """Start the FastAPI webapp."""
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
