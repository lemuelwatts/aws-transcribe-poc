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

"""Meeting Combiner Service.

This module provides functionality to combine normalized transcript JSON
and normalized notes JSON into a single unified meeting JSON structure.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CombinedMeeting:
    """Combined meeting output structure containing transcript and notes."""

    job_name: str
    speakers_count: int
    transcript: dict = field(default_factory=dict)
    attendee_notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_name": self.job_name,
            "speakers_count": self.speakers_count,
            "transcript": self.transcript,
            "attendee_notes": self.attendee_notes,
        }


class MeetingCombiner:
    """Service to combine normalized transcript and notes into a single structure."""

    def __init__(self):
        pass

    def validate_transcript(self, data: dict) -> None:
        """Validate that the transcript JSON has the required structure.

        Args:
            data: The normalized transcript JSON data.

        Raises:
            ValueError: If required keys are missing.
        """
        if "job_name" not in data:
            raise ValueError("Invalid transcript format: missing 'job_name' key")

        if "speakers_count" not in data:
            raise ValueError("Invalid transcript format: missing 'speakers_count' key")

        if "transcript" not in data:
            raise ValueError("Invalid transcript format: missing 'transcript' key")

        if "segments" not in data["transcript"]:
            raise ValueError(
                "Invalid transcript format: missing 'transcript.segments' key"
            )

    def validate_notes(self, data: dict) -> None:
        """Validate that the notes JSON has the required structure.

        Args:
            data: The normalized notes JSON data.

        Raises:
            ValueError: If required keys are missing.
        """
        if "attendee_notes" not in data:
            raise ValueError("Invalid notes format: missing 'attendee_notes' key")

    def combine(self, transcript_data: dict, notes_data: dict) -> CombinedMeeting:
        """Combine normalized transcript and notes into a unified structure.

        Args:
            transcript_data: The normalized transcript JSON data.
            notes_data: The normalized notes JSON data.

        Returns:
            CombinedMeeting with transcript and notes merged.

        Raises:
            ValueError: If either input has invalid structure.
        """
        self.validate_transcript(transcript_data)
        self.validate_notes(notes_data)

        combined = CombinedMeeting(
            job_name=transcript_data["job_name"],
            speakers_count=transcript_data["speakers_count"],
            transcript=transcript_data["transcript"],
            attendee_notes=notes_data["attendee_notes"],
        )

        logger.info(
            f"Combined meeting '{combined.job_name}': "
            f"{len(combined.transcript.get('segments', []))} segments, "
            f"{len(combined.attendee_notes)} attendees"
        )

        return combined
