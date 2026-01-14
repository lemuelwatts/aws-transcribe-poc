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

"""Meeting processing router for combining transcript and notes into a single JSON."""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..services.meeting_combiner import MeetingCombiner

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/combined_meetings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/meeting", tags=["meeting"])


class CombinedMeetingResponse(BaseModel):
    """Response model for combined meeting."""

    success: bool
    job_name: str
    speakers_count: int
    segments_count: int
    attendees_count: int
    saved_json_path: str


@router.post("/combine", response_model=CombinedMeetingResponse)
async def combine_meeting(
    transcript_file: UploadFile = File(...),
    notes_file: UploadFile = File(...),
) -> CombinedMeetingResponse:
    """Combine normalized transcript and notes JSON files into a single meeting JSON.

    Accepts two JSON files:
    - transcript_file: A normalized transcript JSON (from /transcript/normalize)
    - notes_file: A normalized notes JSON (from /notes/normalize)

    Returns a combined JSON structure with both transcript segments and attendee notes.

    Args:
        transcript_file: The normalized transcript JSON file.
        notes_file: The normalized notes JSON file.

    Returns:
        CombinedMeetingResponse with combined data and saved file path.

    Raises:
        HTTPException: If either file is not valid JSON or missing required keys.
    """
    # Parse transcript JSON
    try:
        transcript_content = await transcript_file.read()
        transcript_data = json.loads(transcript_content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transcript JSON file: {e!s}",
        )
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Transcript file encoding error: {e!s}",
        )

    # Parse notes JSON
    try:
        notes_content = await notes_file.read()
        notes_data = json.loads(notes_content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid notes JSON file: {e!s}",
        )
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Notes file encoding error: {e!s}",
        )

    try:
        combiner = MeetingCombiner()
        result = combiner.combine(transcript_data, notes_data)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{result.job_name}_combined_{timestamp}"
        json_path = OUTPUT_DIR / f"{base_filename}.json"

        # Build and save JSON data
        json_data = result.to_dict()

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved combined meeting JSON to {json_path}")

        return CombinedMeetingResponse(
            success=True,
            job_name=result.job_name,
            speakers_count=result.speakers_count,
            segments_count=len(result.transcript.get("segments", [])),
            attendees_count=len(result.attendee_notes),
            saved_json_path=str(json_path),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
