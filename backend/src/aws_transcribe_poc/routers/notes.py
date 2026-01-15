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

"""Notes processing router for normalizing meeting notes into JSON format."""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..services.notes_normalizer import NotesNormalizer

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/normalized_notes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/notes", tags=["notes"])


class NormalizedNotesResponse(BaseModel):
    """Response model for normalized notes."""

    success: bool
    attendees_count: int
    saved_json_path: str
    attendee_notes: dict[str, dict[str, str]]


@router.post("/normalize", response_model=NormalizedNotesResponse)
async def normalize_notes(
    file: UploadFile = File(...),
) -> NormalizedNotesResponse:
    """Normalize a meeting notes text file into attendee-grouped JSON.

    Accepts a plain text file containing meeting notes and returns a normalized
    JSON format where notes are organized by attendee.

    Input format:
        Notes should use brackets to mark attendee sections, e.g.:

        [Eli Thompson]
        - Migration timeline: 2 weeks
        - Can skip security review

        [Fran Reyes]
        - Timeline conflict

        If no attendee markers are found, all notes are attributed to "unknown".

    Args:
        file: A plain text file (.txt) containing meeting notes.

    Returns:
        NormalizedNotesResponse with notes organized by attendee.

    Raises:
        HTTPException: If the file cannot be read or decoded.
    """
    try:
        content = await file.read()
        text_content = content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"File encoding error. Please provide a UTF-8 encoded text file: {e!s}",
        )

    normalizer = NotesNormalizer()
    result = normalizer.normalize(text_content)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(file.filename).stem if file.filename else "notes"
    base_filename = f"{original_name}_normalized_{timestamp}"
    json_path = OUTPUT_DIR / f"{base_filename}.json"

    # Build and save JSON data
    json_data = result.to_dict()

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved normalized notes JSON to {json_path}")

    return NormalizedNotesResponse(
        success=True,
        attendees_count=len(result.attendee_notes),
        saved_json_path=str(json_path),
        attendee_notes=json_data["attendee_notes"],
    )
