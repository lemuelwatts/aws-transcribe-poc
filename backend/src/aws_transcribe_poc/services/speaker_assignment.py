import re
from typing import Optional
import boto3
import json
import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SpeakerMapping(BaseModel):

    speaker_map : dict[str, str] = Field(
        description="",
        examples={"spk_0" : "John Smith", "spk_1": "Jane Doe"}
    )


class SpeakerAssignment():
    # initializer
    def __init__(self):
        self.region = os.environ.get("AWS_REGION", "us-east-1")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=self.region)
        self.model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

    # helper functions #
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from response, removing markdown fences and extra text.
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
                speaker_samples[speaker] = [] # intialize empty list

            speaker_samples[speaker].append(text)

        return speaker_samples

    # core class functions #
    def generate_mapping(self, transcript_data, fix_instructions: str = None) -> dict[str, str]:
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
                "messages": [{
                    "role" : "user",
                    "content": [{"text": task}],
                }],
                "inferenceConfig": {
                    "maxTokens": 300,
                    "temperature": 0,
                    "topP": .9,
                },
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id, 
                body=json.dumps(request_body)
            )
            model_response = json.loads(response["body"].read())
            response_text = model_response["output"]["message"]["content"][0]["text"]

            json_text = self._extract_json(response_text)
            mapping = json.loads(json_text)

            logger.info(f"Generated mapping: {mapping}")
            return mapping


        except Exception as e:
            logger.error(f'Error building request and invoking model: {str(e)}')
            return {}        

    def verify_mapping(self, mapping, transcript_data):

        task = f"""
        **Task**
        Your task is to verify if this speaker mapping is accurate.

        **Mapping to verify**
        {json.dumps(mapping, indent=2)}

        **Transcript Data**
        {json.dumps(transcript_data, indent=2)}

        **Output**
        Please only return JSON
        {{
            "issues": ["list issues you found, if any"],
            "suggestions": ["specific suggestions to fix issues stated"]
        }}

        If no issues found, return {{"issues": [], "suggestions": []}}
        """
        try:
            request_body = {
                "messages": [{
                    "role" : "user",
                    "content": [{"text": task}],
                }],
                "inferenceConfig": {
                    "maxTokens": 400,
                    "temperature": 0,
                    "topP": .9 # instead of considering all possible next tokens, only consider tokens whose cumulative probability adds up to topP
                    # .9 filters out unlikely ones but considers wider range of tokens than lower values
                }
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            model_response = json.loads(response['body'].read())
            response_text = model_response['output']['message']['content'][0]['text']

            json_text = self._extract_json(response_text)

            result = json.loads(json_text)

            return result

        except Exception as e:
            logger.error(f'Error in verifying speaker mapping: {str(e)}')
            raise