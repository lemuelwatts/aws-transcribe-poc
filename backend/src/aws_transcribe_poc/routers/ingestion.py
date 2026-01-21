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

"""Ingestion router for meeting videos and notes upload."""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..services.ingestion import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class NoteUploadResponse(BaseModel):
    """Response model for a single uploaded note."""

    original_filename: str
    normalized_filename: str
    s3_uri: str


class IngestionResponse(BaseModel):
    """Response model for meeting ingestion."""

    meeting_id: str
    video_s3_uri: str
    video_normalized_filename: str
    notes_s3_uris: list[NoteUploadResponse]
    processing_metrics: dict


@router.post("/meeting", response_model=IngestionResponse)
async def ingest_meeting(
    video: UploadFile = File(..., description="Meeting video file"),
    notes: list[UploadFile] = File(
        default=[], description="Optional meeting notes (.txt files)"
    ),
) -> IngestionResponse:
    """Ingest a meeting video and optional notes files.

    Accepts a video file and optional note files, normalizes filenames,
    converts video to WAV, and uploads all files to S3 under an organized
    folder structure:

    ```
    input/<meeting_id>/
      ├── video/<normalized_video_name>.wav
      └── notes/<normalized_note_name>.txt
    ```

    The meeting_id is derived from the normalized video filename plus a timestamp.

    Args:
        video: Meeting video file (mp4, webm, mp3, wav, etc.)
        notes: Optional list of note files (.txt format)

    Returns:
        IngestionResponse with meeting_id and S3 URIs for all uploaded files.

    Raises:
        HTTPException: If video format is unsupported or upload fails.
    """
    tmp_video_path = None
    tmp_note_paths = []

    try:
        # Save video to temp file
        video_suffix = Path(video.filename).suffix if video.filename else ".mp4"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=video_suffix
        ) as tmp_video:
            content = await video.read()
            tmp_video.write(content)
            tmp_video_path = tmp_video.name

        # Save notes to temp files
        notes_data = []
        for note in notes:
            if note.filename:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt"
                ) as tmp_note:
                    note_content = await note.read()
                    tmp_note.write(note_content)
                    tmp_note_paths.append(tmp_note.name)
                    notes_data.append((tmp_note.name, note.filename))

        # Run ingestion
        service = IngestionService()
        result = service.ingest_meeting(
            video_path=tmp_video_path,
            video_filename=video.filename or "meeting_video",
            notes=notes_data if notes_data else None,
        )

        # Build response
        notes_responses = [
            NoteUploadResponse(
                original_filename=nr.original_filename,
                normalized_filename=nr.normalized_filename,
                s3_uri=nr.s3_uri,
            )
            for nr in result.notes_results
        ]

        return IngestionResponse(
            meeting_id=result.meeting_id,
            video_s3_uri=result.video_s3_uri,
            video_normalized_filename=result.video_normalized_filename,
            notes_s3_uris=notes_responses,
            processing_metrics=result.processing_metrics,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp files
        if tmp_video_path and os.path.exists(tmp_video_path):
            try:
                os.unlink(tmp_video_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp video: {e!s}")

        for tmp_path in tmp_note_paths:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp note: {e!s}")
