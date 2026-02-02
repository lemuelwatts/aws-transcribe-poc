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

import json
import logging
import os
import re

import boto3
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpeakerMapping(BaseModel):
    speaker_map: dict[str, str] = Field(
        description="", examples={"spk_0": "John Smith", "spk_1": "Jane Doe"}
    )


class SpeakerAssignment:
    # initializer
    def __init__(self):
        self.region = os.environ.get("AWS_REGION", "us-east-1")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=self.region)
        self.model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

    # helper functions #
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response, removing markdown fences and extra text.
        Sometimes the LLM doesn't respond with pure JSON, it might add
        markdown, whitespace, newlines, explanation.
        """
        if not text:
            return text

        fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(fence_pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        brace_match = re.search(r"(\{[\s\S]*?\})", text)
        if brace_match:
            return brace_match.group(1).strip()

        return text.strip()

    def _build_speaker_samples(self, transcript_data: dict) -> dict:
        """Extract speaker samples from transcript for identification

        Current approach sends all segments from transcript.
        Future optimization consideration: for longer meetings, we may
        not want to send entire transcript and instead take first X amount
        of segments and other strategic samples so that we send less to the model.
        there may be a better approach.

        Args:
            transcript_data: Full transcript dict with "segments" key

        Returns:
            Dict mapping speaker labels to lists of text segments
            Example: {"spk_0": ["text1", "text2"], "spk_1": ["text1"]}
        """
        speaker_samples = {}

        transcript = transcript_data.get("transcript", [])

        segments = transcript.get("segments", [])

        for segment in segments:
            speaker = segment.get("speaker")
            text = segment.get("text", "").strip()

            if speaker not in speaker_samples:
                speaker_samples[speaker] = []  # intialize empty list

            speaker_samples[speaker].append(text)

        return speaker_samples

    # core class functions #
    def generate_mapping(
        self, transcript_data, fix_instructions: str = None
    ) -> dict[str, str]:
        """Generate a mapping of speakers given a transcript.

        Prompt an LLM to evaluate a transcript and use reasoning and inference
        to generate a mapping of the identified speakers from diarization and
        assign them a name based on clues in the transcript.

        Returns a dictionary of :
        {diarized speaker id : speaker name identified by LLM}
        """
        # extract speaker samples
        speaker_samples = self._build_speaker_samples(transcript_data)

        if not speaker_samples:
            logger.warning("No speaker samples found in transcript")
            return {}

        # set up prompt with instructions for model
        task = f"""
        **Task**
        Generate a speaker mapping that identifies real names from diarized speaker labels
        **Context**
        - Environment: Tech company daily standup meetings
        - Participants: Team members discussing projects
        - Data format: Diarized transcript with speaker labels and text

        ** Data to analyze **
        - Input data:
        {json.dumps(speaker_samples, indent=2)}

        {f"Additional Hints: {fix_instructions}" if fix_instructions else ""}
        ** Instructions - How to identify speakers **
        1. Check for self introductions
        2. Look for direct address
        3. Pay attention to who is assigned tasks
        4. Pay attention for who replies after name is called
        5. Use context clues to identify roles, topics, meeting flow
        6. Use process of elimination to narrow down the possiblities

        **Required Output from You**
        Return ONLY a valid json object. No explanations, no markdown.

        Example: {{"spk_0": "Sarah", "spk_1": "Michael", "spk_2": "Emma"}}

        If you cannot identify ANY speakers confidently, return: {{}}

        """
        # pass in entire transcript
        try:
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": task}],
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 300,
                    "temperature": 0,
                    "topP": 0.9,
                },
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id, body=json.dumps(request_body)
            )
            model_response = json.loads(response["body"].read())
            response_text = model_response["output"]["message"]["content"][0]["text"]

            json_text = self._extract_json(response_text)
            mapping = json.loads(json_text)

            logger.info(f"Generated mapping: {mapping}")
            return mapping

        except Exception as e:
            logger.error(f"Error building request and invoking model: {e!s}")
            return {}

    def verify_mapping(self, mapping, transcript_data):
        """Simple verification to check if missing speakers or duplicate name assignment."""
        issues = []

        # check for duplicates
        names = list(mapping.values())
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            issues.append(f"Duplicate names: {set(duplicates)}")

        # check for missing names
        transcript = transcript_data.get("transcript", {})
        segments = transcript.get("segments", [])
        actual_speakers = set(seg.get("speaker") for seg in segments)
        mapped_speakers = set(mapping.keys())
        missing = actual_speakers - mapped_speakers
        if missing:
            issues.append(f"Missing speakers: {missing}")

        return {"issues": issues, "should_retry": len(issues) > 0}
