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
    mapping = assigner.generate_mapping(transcript_data=transcript_data, fix_instructions=hints)

    return IdentifyResponseModel(
        success=bool(mapping),
        speaker_mapping=mapping
    )