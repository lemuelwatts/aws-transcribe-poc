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

from datetime import datetime
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from ..services.ffmpeg_handler import FfmpegHandler
from ..services.speaker_assignment import SpeakerAssignment
from ..services.speaker_identification import SpeakerIdentification
from ..services.storage_manager import StorageManager
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/speakers", tags=["speakers"])


# LLM BASED SPEAKER IDENTIFICATION
class IdentifyRequestModel(BaseModel):
    transcript_file_path: str = Field(
        description="Path to transcript JSON file with segments"
    )
    hints: str = Field(
        default=None,
        description="Hints provided if verification failed and returns corrections.",
    )


class IdentifyResponseModel(BaseModel):
    success: bool
    speaker_mapping: dict[str, str] = Field(
        description="Mapping of speaker labels to names"
    )
    verification: dict = Field(default_factory=dict)
    attempts: int = Field(default=1)


@router.post("/identify", response_model=IdentifyResponseModel)
async def identify_speakers(request: IdentifyRequestModel) -> IdentifyResponseModel:
    """Identify speakers from a transcript

    Args:
        request: identification request with transcript path

    Returns:
        speaker mapping
    """
    try:
        transcript_path = Path(request.transcript_file_path)
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Transcript file path not found: {request.transcript_file_path}"
            )

        with open(transcript_path) as f:
            transcript_data = json.load(f)

        return await _identify_text_based(transcript_data, request.hints)
    except Exception as e:
        return f"Error: {e!s}"


async def _identify_text_based(
    transcript_data: dict, hints: str | None = None
) -> IdentifyResponseModel:
    """Identify speakers using LLM text analysis"""
    logger.info("Using text based speaker identification")

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

        mapping = assigner.generate_mapping(
            transcript_data=transcript_data, fix_instructions=retry_hints
        )
        if not mapping:
            logger.warning("Empty mapping generated")
            break

        verification = assigner.verify_mapping(mapping, transcript_data)
        logger.info(f"verification is :{verification}")

        issues = verification.get("issues", [])
        should_retry = verification.get("should_retry", False)

        if not issues or not should_retry:
            logger.info("No critical issues found")
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
        verification=verification or {"issues": [], "should_retry": False},
        attempts=attempt,
    )


## BIOMETRIC SPEAKER IDENTIFICATION with pyannote/wespeaker-voxceleb-resnet34-LM model


class RegisterVoiceprintRequest(BaseModel):
    name: str
    metadata: dict = {}


class RegisterVoiceprintResponse(BaseModel):
    success: bool
    name: str
    message: str


class IdentifyResponse(BaseModel):
    matched_speaker: str | None
    confidence: float | None


class VoiceprintsResponse(BaseModel):
    speakers: list[str]


class DeleteResponse(BaseModel):
    success: bool
    name: str


@router.post("/register-voiceprint", response_model=RegisterVoiceprintResponse)
async def register_voiceprint(
    name: str,
    file: UploadFile = File(...),
    metadata: str = "{}",
):
    """Register a speaker's voiceprint"""
    try:
        try:
            metadata_dict = json.loads(metadata)
            if not isinstance(metadata_dict, dict):
                raise ValueError("Metadata must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return RegisterVoiceprintResponse(
                success=False,
                name=name,
                message=f"Invalid metadata: {str(e)}"
            )
        
        temp_dir = Path("/tmp/voiceprints")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / file.filename
        
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        
        if temp_file.suffix.lower() == ".wav":
            wav_file = temp_file
            logger.info(f"File is already WAV format: {wav_file}")
        else:
            ffmpeg = FfmpegHandler()
            wav_filename, _ = ffmpeg.convert_to_wav(str(temp_file))
            wav_file = Path(wav_filename)  # convert_to_wav returns filename in CWD
            logger.info(f"Converted to WAV: {wav_file}")

        speaker_id_service = SpeakerIdentification()
        embedding = speaker_id_service.generate_embedding(str(wav_file))

        speaker_metadata = {
            "name": name,
            "date": datetime.now().isoformat(),
            "audio_file_path": str(wav_file),
        }
        speaker_metadata.update(metadata_dict)

        storage_mgr = StorageManager()
        success = storage_mgr.save_speaker(
            speaker_id=name, 
            embedding=embedding, 
            metadata=speaker_metadata
        )
        
        message = (
            f"Successfully registered voiceprint for {name}" if success 
            else f"Failed to save voiceprint for {name}"
        )
        
        return RegisterVoiceprintResponse(
            success=success,
            name=name,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error registering voiceprint: {e}")
        return RegisterVoiceprintResponse(
            success=False,
            name=name,
            message=f"Error: {str(e)}"
        )
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        if wav_file and wav_file.exists() and wav_file != temp_file:
            try:
                wav_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup WAV file {wav_file}: {e}")        


@router.post("/identify-by-voiceprint", response_model=IdentifyResponse)
async def identify_by_voiceprint(
    file: UploadFile = File(...), 
    threshold: float = 0.85
):
    """Identify a speaker from an audio sample"""
    temp_file = None
    wav_file = None
    
    try:
        temp_dir = Path("/tmp/voiceprints")
        temp_dir.mkdir(exist_ok=True, parents=True)
        temp_file = temp_dir / file.filename
        
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        if temp_file.suffix.lower() == ".wav":
            wav_file = temp_file
        else:
            ffmpeg = FfmpegHandler()
            wav_filename, _ = ffmpeg.convert_to_wav(
                str(temp_file),
                output_dir=str(temp_dir)
            )
            wav_file = Path(wav_filename)
        
        storage_mgr = StorageManager()
        stored_embeddings = storage_mgr.get_all_embeddings()
        
        if not stored_embeddings:
            return IdentifyResponse(
                matched_speaker=None,
                confidence=None
            )
        
        speaker_id_service = SpeakerIdentification()
        matched_name, similarity = speaker_id_service.find_match(
            str(wav_file),
            stored_embeddings,
            threshold
        )
        
        return IdentifyResponse(
            matched_speaker=matched_name,
            confidence=float(similarity) if similarity else None
        )
        
    except Exception as e:
        logger.exception(f"Error identifying voiceprint: {e}")
        return IdentifyResponse(
            matched_speaker=None,
            confidence=None
        )
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")
        
        if wav_file and wav_file.exists() and wav_file != temp_file:
            try:
                wav_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup {wav_file}: {e}")


@router.get("/voiceprints", response_model=VoiceprintsResponse)
async def get_voiceprints():
    """List all registered speakers"""
    try:
        storage_manager = StorageManager()
        speakers = storage_manager.list_speakers()
        return VoiceprintsResponse(speakers=speakers)
        
    except Exception as e:
        logger.exception(f"Error listing voiceprints: {e}")
        return VoiceprintsResponse(speakers=[])


@router.delete("/voiceprint/{name}", response_model=DeleteResponse)
async def delete_voiceprint(name: str):
    """Delete a speaker's voiceprint"""
    try:
        storage_manager = StorageManager()
        success = storage_manager.delete_speaker(name)
        
        return DeleteResponse(success=success, name=name)
        
    except Exception as e:
        logger.exception(f"Error deleting voiceprint '{name}': {e}")
        return DeleteResponse(success=False, name=name)