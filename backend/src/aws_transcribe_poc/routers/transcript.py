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

"""Transcript processing router for normalizing AWS Transcribe output."""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..services.transcript_normalizer import TranscriptNormalizer

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/normalized_transcripts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


router = APIRouter(prefix="/transcript", tags=["transcript"])


class NormalizedTranscriptResponse(BaseModel):
    """Response model for normalized transcript."""

    success: bool
    job_name: str
    speakers_count: int
    segments_count: int
    saved_file_path: str


@router.post("/normalize", response_model=NormalizedTranscriptResponse)
async def normalize_transcript(
    file: UploadFile = File(...),
) -> NormalizedTranscriptResponse:
    """Normalize an AWS Transcribe JSON file into speaker-grouped segments.

    Accepts an AWS Transcribe output JSON file and returns a normalized
    format where consecutive speech from the same speaker is merged into
    single segments.

    Args:
        file: The AWS Transcribe JSON output file.

    Returns:
        NormalizedTranscriptResponse with segments organized by speaker
        in chronological order.

    Raises:
        HTTPException: If the file is not valid JSON or missing required keys.
    """
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON file: {e!s}",
        )
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"File encoding error: {e!s}",
        )

    try:
        normalizer = TranscriptNormalizer()
        result = normalizer.normalize(data)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{result.job_name}_normalized_{timestamp}.json"
        output_path = OUTPUT_DIR / output_filename

        # Build response data
        response_data = {
            "job_name": result.job_name,
            "speakers_count": result.speakers_count,
            "segments": [
                {
                    "speaker": seg.speaker,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                }
                for seg in result.segments
            ],
        }

        # Save to file
        with open(output_path, "w") as f:
            json.dump(response_data, f, indent=2)

        logger.info(f"Saved normalized transcript to {output_path}")

        return NormalizedTranscriptResponse(
            success=True,
            job_name=result.job_name,
            speakers_count=result.speakers_count,
            segments_count=len(result.segments),
            saved_file_path=str(output_path),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
