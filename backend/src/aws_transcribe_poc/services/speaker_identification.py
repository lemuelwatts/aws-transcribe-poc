import numpy as np
import logging
from typing import Optional
from pyannote.audio import Inference, Model
from scipy.spatial.distance import cosine
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class SpeakerIdentification:
    def __init__(self, storage_manager=None):
        model_path = Path(__file__).parent.parent / "models" / "wespeaker-voxceleb-resnet34-LM"
        self.model = Model.from_pretrained(str(model_path), local_files_only=True)

        self.inference = Inference(
            self.model,
            window="whole"
        )
    
    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from an audio file.
        
        Args:
            audio_path: Path to the audio file (.wav format)
            
        Returns:
            NumPy array containing the speaker embedding
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If embedding generation fails
        """
        audio_path_obj = Path(audio_path)
        
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            embedding = self.inference(audio_path)
            logger.info(f"Generated embedding for {audio_path}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for {audio_path}: {e}")
            raise

    def store_speaker(
        self, 
        name: str, 
        audio_path: str, 
        metadata: Optional[dict] = None
    ) -> bool:
        """Store a speaker's embedding generated from an audio file.
        
        Args:
            name: Name/identifier for the speaker
            audio_path: Path to the audio file containing speaker's voice
            metadata: Optional additional metadata dictionary
            
        Returns:
            True if speaker was successfully stored, False otherwise
        """
        try:
            embedding = self.generate_embedding(audio_path)
            
            speaker_metadata = {
                'name': name,
                'date': datetime.now().isoformat(),
                'audio_file_path': audio_path
            }
            
            if metadata:
                speaker_metadata.update(metadata)
            
            success = self.storage_manager.save_speaker(
                speaker_id=name,
                embedding=embedding,
                metadata=speaker_metadata
            )
            
            if success:
                logger.info(f"Successfully stored speaker '{name}'")
            else:
                logger.error(f"Failed to store speaker '{name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing speaker '{name}': {e}")
            return False
    
    def get_speaker(self, name: str) -> Optional[dict]:
        """Retrieve a speaker's embedding and metadata.
        
        Args:
            name: Name/identifier of the speaker
            
        Returns:
            Dictionary with 'embedding' and 'metadata' keys, or None if not found
        """
        return self.storage_manager.load_speaker(name)
    
    def compare_speakers(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        threshold: float = 0.85
    ) -> bool:
        """Compare two speaker embeddings to determine if they're the same speaker.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            threshold: Similarity threshold (default: 0.85)
            
        Returns:
            True if speakers are considered the same (similarity >= threshold)
        """
        similarity = 1 - cosine(embedding1, embedding2)
        is_match = similarity >= threshold
        
        logger.debug(f"Speaker comparison: similarity={similarity:.3f}, match={is_match}")
        return is_match
    
    def find_match(
        self, 
        audio_path: str, 
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """Find the best matching speaker for a given audio file.
        
        This method:
        1. Generates an embedding from the input audio file
        2. Compares it against all stored speaker embeddings
        3. Returns the name of the best match if above threshold
        
        Args:
            audio_path: Path to the audio file to identify
            threshold: Optional similarity threshold (uses default if not provided)
            
        Returns:
            Name of the best matching speaker, or None if no match found
        """
        if threshold is None:
            threshold = self.default_threshold
        
        try:
            input_embedding = self.generate_embedding(audio_path)
            
            all_embeddings = self.storage_manager.get_all_embeddings()
            
            if not all_embeddings:
                logger.warning("No speakers stored in database")
                return None
            
            best_match_name = None
            best_similarity = -1.0
            
            for speaker_name, stored_embedding in all_embeddings.items():
                similarity = 1 - cosine(input_embedding, stored_embedding)
                
                logger.debug(f"Comparing with '{speaker_name}': similarity={similarity:.3f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_name = speaker_name
            
            # Check if best match exceeds threshold
            if best_similarity >= threshold:
                logger.info(
                    f"Match found: '{best_match_name}' "
                    f"(similarity: {best_similarity:.3f})"
                )
                return best_match_name
            else:
                logger.info(
                    f"No match found. Best similarity was {best_similarity:.3f} "
                    f"with '{best_match_name}', below threshold {threshold}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Error finding match for {audio_path}: {e}")
            return None
    
    def list_all_speakers(self) -> list[str]:
        """Get a list of all stored speaker names.
        
        Returns:
            List of speaker name strings
        """
        return self.storage_manager.list_speakers()

def main():
    speaker_id = SpeakerIdentification()
    # Store a new speaker
    success = speaker_id.store_speaker(
        name="John Doe",
        audio_path="/path/to/john_sample.wav",
        metadata={"team": "engineering", "role": "developer"}
    )

    print(success)

    # Find a match for an unknown audio file
    matched_speaker = speaker_id.find_match("/path/to/unknown_voice.wav")
    if matched_speaker:
        print(f"Identified speaker: {matched_speaker}")
    else:
        print("No matching speaker found")

    # Get a specific speaker's data
    speaker_data = speaker_id.get_speaker("John Doe")
    if speaker_data:
        print(f"Name: {speaker_data['metadata']['name']}")
        print(f"Date added: {speaker_data['metadata']['date']}")

    # List all speakers
    all_speakers = speaker_id.list_all_speakers()
    print(f"Known speakers: {all_speakers}")

if __name__ == "__main__":
    main()
