from pathlib import Path
from typing import Optional
import numpy as np
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages local storage of speaker embeddings using .npz files."""
    
    def __init__(self, storage_path: str = "data/embeddings"):
        """Initialize storage manager with a base storage path.
        
        Args:
            storage_path: Directory path where embeddings will be stored
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"StorageManager initialized with path: {self.storage_path}")
    
    def _get_speaker_path(self, speaker_id: str) -> Path:
        """Generate file path for a speaker's embedding file.
        
        Args:
            speaker_id: Unique identifier for the speaker
            
        Returns:
            Path object for the speaker's .npz file
        """
        # Sanitize speaker_id to create valid filename
        safe_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in speaker_id)
        return self.storage_path / f"{safe_id}.npz"
    
    def save_speaker(
        self, 
        speaker_id: str, 
        embedding: np.ndarray, 
        metadata: dict
    ) -> bool:
        """Save speaker embedding and metadata to disk.
        
        Args:
            speaker_id: Unique identifier for the speaker
            embedding: NumPy array containing the speaker embedding
            metadata: Dictionary containing speaker metadata (name, date, audio_path, etc.)
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            file_path = self._get_speaker_path(speaker_id)
            
            # Save embedding and metadata in a single .npz file
            np.savez(
                file_path,
                embedding=embedding,
                metadata=np.array([json.dumps(metadata)])  # Store as JSON string
            )
            
            logger.info(f"Successfully saved speaker '{speaker_id}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save speaker '{speaker_id}': {e}")
            return False
    
    def load_speaker(self, speaker_id: str) -> Optional[dict]:
        """Load speaker embedding and metadata from disk.
        
        Args:
            speaker_id: Unique identifier for the speaker
            
        Returns:
            Dictionary containing 'embedding' and 'metadata' keys, or None if not found
        """
        try:
            file_path = self._get_speaker_path(speaker_id)
            
            if not file_path.exists():
                logger.warning(f"Speaker '{speaker_id}' not found at {file_path}")
                return None
            
            data = np.load(file_path, allow_pickle=True)
            metadata = json.loads(str(data['metadata'][0]))
            
            return {
                'embedding': data['embedding'],
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to load speaker '{speaker_id}': {e}")
            return None
    
    def list_speakers(self) -> list[str]:
        """Get list of all stored speaker IDs.
        
        Returns:
            List of speaker ID strings
        """
        try:
            return [f.stem for f in self.storage_path.glob("*.npz")]
        except Exception as e:
            logger.error(f"Failed to list speakers: {e}")
            return []
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker's embedding file.
        
        Args:
            speaker_id: Unique identifier for the speaker
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            file_path = self._get_speaker_path(speaker_id)
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Successfully deleted speaker '{speaker_id}'")
                return True
            else:
                logger.warning(f"Speaker '{speaker_id}' not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete speaker '{speaker_id}': {e}")
            return False
    
    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Load all speaker embeddings.
        
        Returns:
            Dictionary mapping speaker_id to embedding arrays
        """
        embeddings = {}
        
        for speaker_id in self.list_speakers():
            speaker_data = self.load_speaker(speaker_id)
            if speaker_data:
                embeddings[speaker_id] = speaker_data['embedding']
        
        return embeddings