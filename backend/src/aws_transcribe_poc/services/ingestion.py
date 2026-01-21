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

"""Ingestion Service for meeting videos and notes.

This module provides functionality to ingest meeting video files and notes,
normalize filenames, convert video to WAV, and upload to S3 under an
organized folder structure.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .ffmpeg_handler import FfmpegHandler
from .s3_handler import S3Handler

logger = logging.getLogger(__name__)

ACCEPTABLE_VIDEO_TYPES = {
    ".mp3",
    ".mp4",
    ".wav",
    ".m4a",
    ".flac",
    ".avi",
    ".webm",
}


def normalize_filename(name: str) -> str:
    """Normalize a filename by replacing special characters with underscores.

    Args:
        name: The original filename (with or without extension).

    Returns:
        Normalized filename with only alphanumeric, hyphens, and underscores.
    """
    # Get the stem (filename without extension)
    stem = Path(name).stem

    # Keep only alphanumeric, hyphens, and underscores
    normalized = re.sub(r"[^\w\-]", "_", stem)

    # Replace multiple underscores with single underscore
    normalized = re.sub(r"_+", "_", normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip("_")

    return normalized


@dataclass
class NoteUploadResult:
    """Result of a single note file upload."""

    original_filename: str
    normalized_filename: str
    s3_uri: str
    metrics: dict


@dataclass
class IngestionResult:
    """Result of meeting ingestion."""

    meeting_id: str
    video_s3_uri: str
    video_normalized_filename: str
    notes_results: list[NoteUploadResult] = field(default_factory=list)
    processing_metrics: dict = field(default_factory=dict)


class IngestionService:
    """Service to ingest meeting videos and notes into S3."""

    def __init__(self):
        self.ffmpeg_handler = FfmpegHandler()
        self.s3_handler = S3Handler()

    def _check_video_extension(self, file_path: str) -> None:
        """Validate that the file has an acceptable video/audio extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = Path(file_path).suffix.lower()
        if extension not in ACCEPTABLE_VIDEO_TYPES:
            raise ValueError(
                f"Unsupported video type: {extension}. "
                f"Supported types: {ACCEPTABLE_VIDEO_TYPES}"
            )

    def _generate_meeting_id(self, video_filename: str) -> str:
        """Generate a meeting ID from the video filename.

        Args:
            video_filename: The original video filename.

        Returns:
            Normalized meeting ID with timestamp.
        """
        normalized = normalize_filename(video_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{normalized}_{timestamp}"

    def _process_video(
        self, video_path: str, meeting_id: str, video_filename: str
    ) -> tuple[str, str, dict]:
        """Process video file: convert to WAV and upload to S3.

        Args:
            video_path: Path to the video file.
            meeting_id: The meeting identifier for S3 path.
            video_filename: Original video filename.

        Returns:
            Tuple of (s3_uri, normalized_filename, metrics).
        """
        start_time = time.time()
        wav_filepath = None

        try:
            # Validate video extension
            self._check_video_extension(video_path)

            # Probe file for metadata
            probe = self.ffmpeg_handler.probe_file(video_path)
            input_metadata = {
                "format": probe["format"],
                "streams": probe["streams"],
            }

            # Convert to WAV
            wav_filepath, ffmpeg_metrics = self.ffmpeg_handler.convert_to_wav(
                video_path
            )

            # Generate normalized filename
            normalized_name = normalize_filename(video_filename)
            s3_key = f"input/{meeting_id}/video/{normalized_name}.wav"

            # Upload to S3
            s3_uri, s3_metrics = self.s3_handler.upload_file_to_path(
                wav_filepath, s3_key
            )

            total_time = time.time() - start_time

            metrics = {
                "total_processing_time_seconds": round(total_time, 3),
                "input_metadata": input_metadata,
                **ffmpeg_metrics,
                **s3_metrics,
            }

            return s3_uri, normalized_name, metrics

        finally:
            # Clean up temp WAV file
            if wav_filepath and os.path.exists(wav_filepath):
                try:
                    os.unlink(wav_filepath)
                    logger.info(f"Cleaned up temp WAV file: {wav_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {wav_filepath}: {e!s}")

    def _upload_note(
        self, note_path: str, meeting_id: str, note_filename: str
    ) -> NoteUploadResult:
        """Upload a single note file to S3.

        Args:
            note_path: Path to the note file.
            meeting_id: The meeting identifier for S3 path.
            note_filename: Original note filename.

        Returns:
            NoteUploadResult with upload details.
        """
        normalized_name = normalize_filename(note_filename)
        s3_key = f"input/{meeting_id}/notes/{normalized_name}.txt"

        s3_uri, metrics = self.s3_handler.upload_file_to_path(note_path, s3_key)

        return NoteUploadResult(
            original_filename=note_filename,
            normalized_filename=f"{normalized_name}.txt",
            s3_uri=s3_uri,
            metrics=metrics,
        )

    def ingest_meeting(
        self,
        video_path: str,
        video_filename: str,
        notes: list[tuple[str, str]] | None = None,
    ) -> IngestionResult:
        """Ingest a meeting video and optional notes into S3.

        Args:
            video_path: Path to the video file.
            video_filename: Original video filename.
            notes: Optional list of (note_path, note_filename) tuples.

        Returns:
            IngestionResult with meeting_id and S3 URIs.
        """
        start_time = time.time()

        # Generate meeting ID from video filename
        meeting_id = self._generate_meeting_id(video_filename)
        logger.info(f"Ingesting meeting: {meeting_id}")

        # Process video
        video_s3_uri, video_normalized, video_metrics = self._process_video(
            video_path, meeting_id, video_filename
        )

        # Process notes
        notes_results = []
        notes_metrics = []
        if notes:
            for note_path, note_filename in notes:
                try:
                    result = self._upload_note(note_path, meeting_id, note_filename)
                    notes_results.append(result)
                    notes_metrics.append(
                        {
                            "filename": note_filename,
                            "normalized": result.normalized_filename,
                            **result.metrics,
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to upload note {note_filename}: {e!s}")
                    raise

        total_time = time.time() - start_time

        processing_metrics = {
            "total_ingestion_time_seconds": round(total_time, 3),
            "video": video_metrics,
            "notes": notes_metrics,
        }

        logger.info(
            f"Meeting ingestion complete: {meeting_id}, "
            f"video: {video_s3_uri}, notes: {len(notes_results)}"
        )

        return IngestionResult(
            meeting_id=meeting_id,
            video_s3_uri=video_s3_uri,
            video_normalized_filename=f"{video_normalized}.wav",
            notes_results=notes_results,
            processing_metrics=processing_metrics,
        )
