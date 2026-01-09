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

"""Transcript Normalizer Service.

This module provides functionality to normalize AWS Transcribe output
into a structured format organized by speaker in chronological order.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a segment of speech from a single speaker."""

    speaker: str
    start_time: float
    end_time: float
    text: str


@dataclass
class NormalizedTranscript:
    """Normalized transcript output structure."""

    job_name: str
    speakers_count: int
    segments: list[SpeakerSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_name": self.job_name,
            "speakers_count": self.speakers_count,
            "segments": [
                {
                    "speaker": seg.speaker,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                }
                for seg in self.segments
            ],
        }


class TranscriptNormalizer:
    """Service to normalize AWS Transcribe output by speaker."""

    def __init__(self):
        pass

    def validate_input(self, data: dict) -> None:
        """Validate that the input has the required AWS Transcribe structure.

        Args:
            data: The AWS Transcribe JSON data.

        Raises:
            ValueError: If required keys are missing.
        """
        if "results" not in data:
            raise ValueError("Invalid AWS Transcribe format: missing 'results' key")

        results = data["results"]

        if "items" not in results:
            raise ValueError("Invalid AWS Transcribe format: missing 'results.items'")

        if "speaker_labels" not in results:
            raise ValueError(
                "Invalid AWS Transcribe format: missing 'results.speaker_labels'. "
                "Ensure speaker identification was enabled for the transcription job."
            )

    def normalize(self, data: dict) -> NormalizedTranscript:
        """Normalize AWS Transcribe output into speaker-grouped segments.

        Groups consecutive speech from the same speaker into single segments.
        When the speaker changes, a new segment begins.

        Args:
            data: The AWS Transcribe JSON data.

        Returns:
            NormalizedTranscript with segments organized by speaker.
        """
        self.validate_input(data)

        job_name = data.get("jobName", "unknown")
        results = data["results"]
        items = results["items"]
        speaker_labels = results["speaker_labels"]
        speakers_count = speaker_labels.get("speakers", 0)

        segments: list[SpeakerSegment] = []

        # Track current segment being built
        current_speaker: str | None = None
        current_start_time: float | None = None
        current_end_time: float | None = None
        current_words: list[str] = []

        for item in items:
            item_type = item.get("type")
            speaker = item.get("speaker_label")
            content = item.get("alternatives", [{}])[0].get("content", "")

            if item_type == "punctuation":
                # Punctuation has no timestamps, attach to current segment
                if current_words:
                    # Append punctuation without space
                    current_words[-1] = current_words[-1] + content
                continue

            # It's a pronunciation item
            start_time = float(item.get("start_time", 0))
            end_time = float(item.get("end_time", 0))

            if current_speaker is None:
                # First word - start new segment
                current_speaker = speaker
                current_start_time = start_time
                current_end_time = end_time
                current_words = [content]
            elif speaker == current_speaker:
                # Same speaker - extend current segment
                current_end_time = end_time
                current_words.append(content)
            else:
                # Speaker changed - finalize current segment and start new one
                if current_words and current_speaker is not None:
                    segments.append(
                        SpeakerSegment(
                            speaker=current_speaker,
                            start_time=current_start_time,
                            end_time=current_end_time,
                            text=" ".join(current_words),
                        )
                    )

                # Start new segment
                current_speaker = speaker
                current_start_time = start_time
                current_end_time = end_time
                current_words = [content]

        # Finalize last segment
        if current_words and current_speaker is not None:
            segments.append(
                SpeakerSegment(
                    speaker=current_speaker,
                    start_time=current_start_time,
                    end_time=current_end_time,
                    text=" ".join(current_words),
                )
            )

        logger.info(
            f"Normalized transcript '{job_name}': {len(segments)} segments, "
            f"{speakers_count} speakers"
        )

        return NormalizedTranscript(
            job_name=job_name,
            speakers_count=speakers_count,
            segments=segments,
        )
