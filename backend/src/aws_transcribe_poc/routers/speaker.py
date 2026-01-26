import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.speaker_assignment import SpeakerAssignment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/speakers", tags=["speakers"])

class IdentifyRequestModel(BaseModel):
    
    transcript_file_path: str = Field(description="Path to transcript JSON file with segments")
    hints: str = Field(default=None, description="Hints provided if verification failed and returns corrections.")

class IdentifyResponseModel(BaseModel):
    
    success: bool
    speaker_mapping: dict[str, str] = Field(
        description="Mapping of speaker labels to names"
    )
    verification: dict = Field(default_factory=dict)
    attempts: int = Field(default=1)

@router.post("/identify", response_model=IdentifyResponseModel)
async def identify_speakers(request: IdentifyRequestModel) -> IdentifyResponseModel:
    """ Identify speakers from a transcript
    
    Args:
        request: identification request with transcript path 
    
    Returns:
        speaker mapping
    """

    try:
        transcript_path = Path(request.transcript_file_path)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file path not found: {request.transcript_file_path}")
        
        with open(transcript_path) as f:
            transcript_data = json.load(f)
        
        return await _identify_text_based(transcript_data, request.hints)
    except Exception as e:
        return f"Error: {str(e)}"

async def _identify_text_based(
    transcript_data: dict,
    hints: str | None = None
) -> IdentifyResponseModel:
    """ Identify speakers using LLM text analysis """
    logger.info('Using text based speaker identification')

    assigner = SpeakerAssignment()
    # add a retry here passing fix instructions if the verify func returned anything
    # ie try block for generate_mapping while verify not empty
    max_retries = 2
    attempt = 0

    mapping = None
    verification = None
    retry_hints = hints

    while attempt < max_retries:
        attempt += 1

        mapping = assigner.generate_mapping(transcript_data=transcript_data, fix_instructions=retry_hints)
        if not mapping:
            logger.warning('Empty mapping generated')
            break

        verification = assigner.verify_mapping(mapping, transcript_data)
        logger.info(f'verification is :{verification}')
        
        issues = verification.get('issues', [])
        should_retry = verification.get('should_retry', False)

        if not issues or not should_retry:
            logger.info('No critical issues found')
            break

        if attempt < max_retries:
            retry_hints = f"""Previous mapping had these issues: {json.dumps(issues, indent=2)}
            {hints if hints else ""}

            Please fix these specific issues in the new mapping.
            """
            logger.warning(f"Critical issues found, retrying with hints: {issues}")
        else:
            logger.warning(f"Max retries reached with issues: {issues}")

    return IdentifyResponseModel(
        success=bool(mapping),
        speaker_mapping=mapping or {},
        verification = verification or {"issues": [], "should_retry": False},
        attempts=attempt
    )