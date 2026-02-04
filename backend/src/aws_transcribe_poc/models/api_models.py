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

"""Pydantic models for the Meeting Processing API."""

from enum import Enum

from pydantic import BaseModel, Field


class SpeakerMethod(str, Enum):
    """Speaker identification methods.

    - NONE: Keep original labels (spk_0, spk_1, etc.)
    - LLM: Infer names from transcript context
    - BIOMETRICS: Match voice against stored voiceprints
    - HYBRID: Both methods, biometrics preferred with LLM fallback
    """

    NONE = "none"
    LLM = "llm"
    BIOMETRICS = "biometrics"
    HYBRID = "hybrid"


class TranscriptionResult(BaseModel):
    """Result from AWS Transcribe."""

    s3_uri: str
    success: bool
    s3_output_uri: str | None = None
    s3_summary_uri: str | None = None
    transcription_time_seconds: float | None = None
    error: str | None = None


class ProcessingMetrics(BaseModel):
    """Audio processing performance metrics."""

    total_time_seconds: float
    ffmpeg_time_seconds: float
    s3_upload_time_seconds: float
    s3_upload_speed_mbps: float
    input_size_mb: float
    output_size_mb: float
    size_reduction_percent: float


class SpeakerMatch(BaseModel):
    """Result of matching a speaker label to a name."""

    name: str
    confidence: str = Field(
        description="Confidence level: high, medium, low, very_low, or inferred (LLM)"
    )
    similarity: float | None = Field(
        None, description="Similarity score 0.0-1.0 (biometrics only)"
    )
    method: str = Field(description="Method used: biometric or llm")


class MeetingResponse(BaseModel):
    """Response from the /process_meeting endpoint."""

    success: bool
    filename: str
    s3_uri: str

    transcription: TranscriptionResult
    processing_metrics: ProcessingMetrics | None = None
    speaker_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Final mapping of speaker labels to names (e.g., {'spk_0': 'Alice'})",
    )
    speaker_details: dict[str, SpeakerMatch] = Field(
        default_factory=dict,
        description="Detailed match info per speaker (confidence, similarity, method)",
    )
    unmatched_speakers: list[str] = Field(
        default_factory=list,
        description="Speaker labels that could not be identified",
    )
    speaker_method_used: str = Field(
        default="none", description="The speaker identification method that was used"
    )
    analysis: dict | None = Field(
        None, description="Meeting analysis report (summary, action items, etc.)"
    )
    analysis_path: str | None = Field(
        None, description="Path where analysis report was saved"
    )
    total_duration_seconds: float
    error: str | None = None
