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

"""AWS Transcribe service for transcription of media files in S3."""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import boto3


@dataclass
class TranscriptionResult:
    """Result of a transcription job."""

    s3_uri: str
    success: bool
    s3_output_uri: str | None = None
    error: str | None = None


class TranscriptionService:
    """Service for transcribing media files using AWS Transcribe."""

    def __init__(self) -> None:
        """Initialize the transcription service."""
        self.region = os.environ.get("AWS_REGION", "us-east-1")

        # Initialize AWS Transcribe client
        self.transcribe_client = boto3.client("transcribe", region_name=self.region)

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse an S3 URI into bucket and key components.

        Args:
            s3_uri: S3 URI (e.g., s3://bucket/key.mp4).

        Returns:
            Tuple of (bucket_name, key).
        """
        # Remove s3:// prefix and split into bucket and key
        path = s3_uri.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def _start_job(self, s3_uri: str, job_name: str) -> str:
        """Start an AWS Transcribe job.

        Args:
            s3_uri: S3 URI of the media file (e.g., s3://bucket/key.mp4).
            job_name: Unique name for the transcription job.

        Returns:
            The job name.
        """
        # Determine media format from file extension
        file_ext = s3_uri.split(".")[-1].lower()

        # Parse S3 URI to get bucket and construct output path
        bucket, key = self._parse_s3_uri(s3_uri)
        filename = key.split("/")[-1]
        output_filename = Path(filename).stem + "_transcription.json"
        output_key = f"output/{output_filename}"

        self.transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat=file_ext,
            LanguageCode="en-US",
            OutputBucketName=bucket,
            OutputKey=output_key,
        )
        return job_name

    def poll_job_status(self, job_name: str, poll_interval: int = 5) -> dict:
        """Poll for transcription job completion.

        Args:
            job_name: The transcription job name.
            poll_interval: Seconds between status checks.

        Returns:
            The completed job response.
        """
        while True:
            response = self.transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                return response
            if status == "FAILED":
                failure_reason = response["TranscriptionJob"].get(
                    "FailureReason", "Unknown error"
                )
                msg = f"Transcription job failed: {failure_reason}"
                raise RuntimeError(msg)

            time.sleep(poll_interval)

    def transcribe_s3_file(self, s3_uri: str) -> TranscriptionResult:
        """Transcribe a media file from S3.

        Args:
            s3_uri: S3 URI of the media file (e.g., s3://bucket/key.mp4).

        Returns:
            TranscriptionResult with status and output path.
        """
        try:
            # Extract filename for job name
            bucket, key = self._parse_s3_uri(s3_uri)
            filename = key.split("/")[-1]
            file_stem = Path(filename).stem

            # Compute S3 output URI (JSON file in output/ folder)
            s3_output_uri = f"s3://{bucket}/output/{file_stem}_transcription.json"

            # Generate unique job name with timestamp (MMDDYYYYHHmmss)
            timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
            job_name = f"transcribe-{file_stem}-{timestamp}"

            # Start transcription job
            self._start_job(s3_uri, job_name)

            # Wait for completion
            self.poll_job_status(job_name)

            return TranscriptionResult(
                s3_uri=s3_uri,
                success=True,
                s3_output_uri=s3_output_uri,
            )

        except Exception as e:
            return TranscriptionResult(
                s3_uri=s3_uri,
                success=False,
                error=str(e),
            )

    def transcribe_all(self, s3_uris: list[str]) -> list[TranscriptionResult]:
        """Transcribe multiple media files from S3.

        Args:
            s3_uris: List of S3 URIs to transcribe.

        Returns:
            List of TranscriptionResult for each file processed.
        """
        results = []

        for s3_uri in s3_uris:
            result = self.transcribe_s3_file(s3_uri)
            results.append(result)

        return results
