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

import json
import os
import re
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
    s3_summary_uri: str | None = None
    s3_metrics_uri: str | None = None
    transcription_duration_seconds: float | None = None
    summary_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    error: str | None = None


class TranscriptionService:
    """Service for transcribing media files using AWS Transcribe."""

    def __init__(self) -> None:
        """Initialize the transcription service."""
        self.region = os.environ.get("AWS_REGION", "us-east-1")

        # Initialize AWS clients
        self.transcribe_client = boto3.client("transcribe", region_name=self.region)
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=self.region)

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

    def _sanitize_job_name(self, name: str) -> str:
        """Sanitize a string for use as AWS Transcribe job name.

        Job names must match pattern: ^[0-9a-zA-Z._-]+

        Args:
            name: The raw name to sanitize.

        Returns:
            Sanitized name with only allowed characters.
        """
        # Replace spaces with underscores
        sanitized = name.replace(" ", "_")
        # Remove any characters that aren't alphanumeric, dot, underscore, or hyphen
        sanitized = re.sub(r"[^0-9a-zA-Z._-]", "", sanitized)
        return sanitized

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

    def _get_transcript_text(self, bucket: str, key: str) -> str:
        """Download transcription JSON from S3 and extract the transcript text.

        Args:
            bucket: S3 bucket name.
            key: S3 object key for the transcription JSON.

        Returns:
            The transcript text.
        """
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        transcription_data = json.loads(response["Body"].read().decode("utf-8"))

        # AWS Transcribe JSON: results.transcripts[0].transcript
        return transcription_data["results"]["transcripts"][0]["transcript"]

    def _summarize_text(self, text: str) -> str:
        """Summarize text using AWS Bedrock with Amazon Titan.

        Args:
            text: The text to summarize.

        Returns:
            The summarized text.
        """
        prompt = (
            "You are an expert meeting summarizer. You MUST produce a summary with "
            "EXACTLY TWO REQUIRED SECTIONS: 'Summary' and 'Action Items'. "
            "These sections are MANDATORY and must ALWAYS be included "
            "with NO EXCEPTIONS.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "Your response must contain both sections below in "
            "this exact structure:\n\n"
            "## Summary\n"
            "[Your summary content here]\n\n"
            "## Action Items\n"
            "[Your action items content here]\n\n"
            "---\n\n"
            "SECTION 1 - SUMMARY (REQUIRED - MUST BE INCLUDED):\n"
            "Write a clear and concise paragraph that captures:\n"
            "- Key topics discussed\n"
            "- Main points raised\n"
            "- Decisions made\n"
            "- Notable conclusions\n"
            "Use complete sentences and professional language. Base your summary "
            "ONLY on information explicitly mentioned in the transcript.\n\n"
            "SECTION 2 - ACTION ITEMS (REQUIRED - MUST BE INCLUDED):\n"
            "Create a comprehensive bulleted list of all action items. For each item:\n"
            "- State the task clearly\n"
            "- Identify who is responsible (or write 'No assignee identified')\n"
            "- Include deadlines/timeframes (or write 'No deadline provided')\n\n"
            "If there are truly no action items in the meeting, you MUST still include "
            "the 'Action Items' section header and write: "
            "'No action items identified.'\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Both sections (Summary AND Action Items) are MANDATORY\n"
            "- NEVER omit either section for any reason\n"
            "- Base content solely on the transcript provided\n"
            "- Maintain a neutral, professional tone\n"
            "- Do not invent or assume information not in the transcript\n\n"
            "---\n\n"
            f"Transcript:\n{text}\n\n"
            "Remember: Your response MUST include both the Summary section and the "
            "Action Items section. Do not skip either one."
        )

        model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-text-express-v1")

        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "temperature": 0.3,
                "topP": 0.9,
            },
        }

        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        return response_body["results"][0]["outputText"]

    def _upload_summary(self, bucket: str, key: str, summary: str) -> None:
        """Upload summary text to S3.

        Args:
            bucket: S3 bucket name.
            key: S3 object key for the summary file.
            summary: The summary text to upload.
        """
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=summary.encode("utf-8"),
            ContentType="text/plain",
        )

    def _upload_metrics(self, bucket: str, key: str, metrics: dict) -> None:
        """Upload metrics JSON to S3.

        Args:
            bucket: S3 bucket name.
            key: S3 object key for the metrics file.
            metrics: The metrics dictionary to upload.
        """
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metrics, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def transcribe_s3_file(
        self, s3_uri: str, *, save_metrics: bool = False
    ) -> TranscriptionResult:
        """Transcribe a media file from S3 and generate a summary.

        Args:
            s3_uri: S3 URI of the media file (e.g., s3://bucket/key.mp4).
            save_metrics: Whether to save metrics JSON to S3 (default: False).

        Returns:
            TranscriptionResult with status, output path, and summary path.
        """
        total_start = time.time()

        try:
            # Extract filename for job name
            bucket, key = self._parse_s3_uri(s3_uri)
            filename = key.split("/")[-1]
            file_stem = Path(filename).stem

            # Compute S3 output URIs
            transcription_key = f"output/{file_stem}_transcription.json"
            summary_key = f"output/{file_stem}_summary.txt"
            s3_output_uri = f"s3://{bucket}/{transcription_key}"
            s3_summary_uri = f"s3://{bucket}/{summary_key}"

            # Generate unique job name with timestamp (YYYYMMDDHHmmss)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            sanitized_stem = self._sanitize_job_name(file_stem)
            job_name = f"transcribe-{sanitized_stem}-{timestamp}"

            # Start transcription job and track duration
            transcription_start = time.time()
            self._start_job(s3_uri, job_name)
            self.poll_job_status(job_name)
            transcription_duration = time.time() - transcription_start

            # Get transcript text and generate summary, track duration
            summary_start = time.time()
            transcript_text = self._get_transcript_text(bucket, transcription_key)
            summary = self._summarize_text(transcript_text)
            self._upload_summary(bucket, summary_key, summary)
            summary_duration = time.time() - summary_start

            total_duration = time.time() - total_start

            # Compute metrics key with completion timestamp
            completion_timestamp = datetime.now()
            completion_timestamp_str = completion_timestamp.strftime("%Y%m%d%H%M%S")
            metrics_key = (
                f"output/metrics/{file_stem}_metrics_{completion_timestamp_str}.json"
            )
            s3_metrics_uri = f"s3://{bucket}/{metrics_key}" if save_metrics else None

            result = TranscriptionResult(
                s3_uri=s3_uri,
                success=True,
                s3_output_uri=s3_output_uri,
                s3_summary_uri=s3_summary_uri,
                s3_metrics_uri=s3_metrics_uri,
                transcription_duration_seconds=round(transcription_duration, 2),
                summary_duration_seconds=round(summary_duration, 2),
                total_duration_seconds=round(total_duration, 2),
            )

            # Upload metrics if enabled
            if save_metrics:
                metrics = {
                    "completed_at": completion_timestamp.isoformat(),
                    "s3_uri": result.s3_uri,
                    "success": result.success,
                    "s3_output_uri": result.s3_output_uri,
                    "s3_summary_uri": result.s3_summary_uri,
                    "s3_metrics_uri": result.s3_metrics_uri,
                    "transcription_duration_seconds": (
                        result.transcription_duration_seconds
                    ),
                    "summary_duration_seconds": result.summary_duration_seconds,
                    "total_duration_seconds": result.total_duration_seconds,
                    "error": result.error,
                }
                self._upload_metrics(bucket, metrics_key, metrics)

            return result

        except Exception as e:
            total_duration = time.time() - total_start
            return TranscriptionResult(
                s3_uri=s3_uri,
                success=False,
                error=str(e),
                total_duration_seconds=round(total_duration, 2),
            )

    def transcribe_all(
        self, s3_uris: list[str], *, save_metrics: bool = False
    ) -> list[TranscriptionResult]:
        """Transcribe multiple media files from S3.

        Args:
            s3_uris: List of S3 URIs to transcribe.
            save_metrics: Whether to save metrics JSON to S3 (default: False).

        Returns:
            List of TranscriptionResult for each file processed.
        """
        results = []

        for s3_uri in s3_uris:
            result = self.transcribe_s3_file(s3_uri, save_metrics=save_metrics)
            results.append(result)

        return results
