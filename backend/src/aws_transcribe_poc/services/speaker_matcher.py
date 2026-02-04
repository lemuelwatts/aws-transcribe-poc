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

"""Speaker Matcher Service.

Identifies speakers in a transcript using three methods:
- LLM: Infer names from transcript context
- Biometrics: Match voice against stored voiceprints
- Hybrid: Both methods, biometrics preferred with LLM fallback
"""

import logging
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..models.api_models import SpeakerMatch, SpeakerMethod
from .ffmpeg_handler import FfmpegHandler
from .speaker_assignment import SpeakerAssignment
from .speaker_identification import SpeakerIdentification
from .storage_manager import StorageManager
from .transcript_normalizer import NormalizedTranscript

logger = logging.getLogger(__name__)


@dataclass
class SpeakerMatchResult:
    """Result of speaker matching operation."""

    mapping: dict[str, str] = field(default_factory=dict)
    details: dict[str, SpeakerMatch] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)


class SpeakerMatcher:
    """Service for identifying speakers in a transcript."""

    def __init__(self):
        self.speaker_id = SpeakerIdentification()
        self.speaker_assign = SpeakerAssignment()
        self.storage = StorageManager()
        self.ffmpeg = FfmpegHandler()

    def match(
        self,
        method: SpeakerMethod,
        audio_path: str,
        normalized_transcript: NormalizedTranscript,
        threshold: float = 0.5,
    ) -> SpeakerMatchResult:
        """Identify speakers in a transcript.

        Args:
            method: Which identification method to use
            audio_path: Path to the audio/video file
            raw_transcript: Raw AWS Transcribe JSON output
            threshold: Similarity threshold for biometric matching (0.0-1.0)

        Returns:
            SpeakerMatchResult with mapping, details, and unmatched speakers
        """
        if method == SpeakerMethod.NONE:
            return SpeakerMatchResult()

        if method == SpeakerMethod.LLM:
            return self._match_llm(normalized_transcript)

        if method == SpeakerMethod.BIOMETRICS:
            return self._match_biometric(audio_path, normalized_transcript, threshold)

        # HYBRID: Run both, prefer biometrics
        return self._match_hybrid(audio_path, normalized_transcript, threshold)

    def _match_llm(self, normalized: NormalizedTranscript) -> SpeakerMatchResult:
        """Infer speaker names from transcript context using LLM."""
        # Get LLM inference
        llm_mapping = self.speaker_assign.generate_mapping(normalized.to_dict())

        # Build result
        result = SpeakerMatchResult()
        expected_speakers = self._extract_speakers(normalized)

        for label, name in llm_mapping.items():
            result.mapping[label] = name
            result.details[label] = SpeakerMatch(
                name=name,
                confidence="inferred",
                similarity=None,
                method="llm",
            )

        # Track unmatched
        result.unmatched = [s for s in expected_speakers if s not in result.mapping]

        logger.info(f"LLM matched {len(result.mapping)} speakers")
        return result

    def _match_biometric(
        self, audio_path: str, normalized: NormalizedTranscript, threshold: float
    ) -> SpeakerMatchResult:
        """Match speakers using voice biometrics against stored voiceprints."""
        result = SpeakerMatchResult()

        speaker_segments = defaultdict(list)
        for seg in normalized.segments:
            speaker_segments[seg.speaker].append(
                (seg.start_time, seg.end_time, seg.text)
            )

        # Get stored voiceprints
        all_speakers = self.storage.get_all_speakers()
        if not all_speakers:
            result.unmatched = list(speaker_segments.keys())
            logger.warning("No stored voiceprints available for biometric matching")
            return result

        # Build embeddings dict
        stored_embeddings = {
            sid: data["embedding"] for sid, data in all_speakers.items()
        }

        # Match each speaker using their longest segment
        with tempfile.TemporaryDirectory(prefix="speaker_match_") as temp_dir:
            for label, segments in speaker_segments.items():
                # Use longest segment to capture voiceprint
                start, end, _ = max(segments, key=lambda s: s[1] - s[0])
                segment_file = Path(temp_dir) / f"{label}.wav"

                try:
                    self.ffmpeg.extract_segment(
                        audio_path, start, end, str(segment_file)
                    )
                    match_id, score = self.speaker_id.find_match(
                        str(segment_file), stored_embeddings, threshold
                    )

                    confidence = self._classify_confidence(score)

                    if match_id and confidence != "none":
                        # Get full name from metadata
                        metadata = all_speakers[match_id].get("metadata", {})
                        name = (
                            metadata.get("full_name")
                            or metadata.get("name")
                            or match_id
                        )

                        result.mapping[label] = name
                        result.details[label] = SpeakerMatch(
                            name=name,
                            confidence=confidence,
                            similarity=round(score, 3),
                            method="biometric",
                        )
                        logger.info(f"Biometric: {label} -> {name} (score={score:.3f})")
                    else:
                        result.unmatched.append(label)

                except Exception as e:
                    logger.error(f"Error matching {label}: {e}")
                    result.unmatched.append(label)

        logger.info(
            f"Biometric matched {len(result.mapping)}, "
            f"unmatched {len(result.unmatched)}"
        )
        return result

    def _match_hybrid(
        self, audio_path: str, raw_transcript: dict, threshold: float
    ) -> SpeakerMatchResult:
        """Run both biometric and LLM, prefer biometric with LLM fallback."""

        bio_result = self._match_biometric(audio_path, raw_transcript, threshold)

        llm_result = self._match_llm(raw_transcript)

        # Merge: biometric preferred, LLM fills gaps
        result = SpeakerMatchResult()
        expected_speakers = self._extract_speakers(raw_transcript)

        for label in expected_speakers:
            if label in bio_result.mapping:
                # Use biometric match
                result.mapping[label] = bio_result.mapping[label]
                result.details[label] = bio_result.details[label]
            elif label in llm_result.mapping:
                # Fall back to LLM
                result.mapping[label] = llm_result.mapping[label]
                result.details[label] = llm_result.details[label]
            else:
                # Neither matched
                result.unmatched.append(label)

        logger.info(
            f"Hybrid: {len(bio_result.mapping)} biometric, "
            f"{len(result.mapping) - len(bio_result.mapping)} LLM fallback, "
            f"{len(result.unmatched)} unmatched"
        )
        return result

    def _extract_speakers(self, normalized: NormalizedTranscript) -> set[str]:
        """Extract unique speaker labels from normalized transcript."""
        return {seg.speaker for seg in normalized.segments}

    def _classify_confidence(self, similarity: float) -> str:
        """Classify confidence level based on similarity score."""
        if similarity >= 0.9:
            return "high"
        if similarity >= 0.8:
            return "medium"
        if similarity >= 0.7:
            return "low"
        if similarity >= 0.5:
            return "very_low"
        return "none"
