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

"""Audio processing router for file upload and processing"""

import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..services.input_handler import InputHandler

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/audio", tags=["audio"])


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


class AudioProcessResponse(BaseModel):
    """Response model for audio processing."""

    s3_uri: str
    original_filename: str
    message: str = "Audio Processed Successfully"
    metrics: PerformanceMetrics


@router.post("/process", response_model=AudioProcessResponse)
async def process_audio(file: UploadFile = File(...)) -> AudioProcessResponse:
    """Upload and process audio/video files

    Converts to WAV format and uploads to s3.

    Args:

    Returns:
        s3 URI
    """
    request_start = time.time()
    # save file temporarily so we can send it across network
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        handler = InputHandler()
        uri, metrics = handler.process_input(tmp_file_path, file.filename)
        logger.info(f"request time: {round(time.time() - request_start, 3)}")
        return AudioProcessResponse(
            s3_uri=uri,
            original_filename=file.filename,
            metrics=PerformanceMetrics(**metrics),
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
