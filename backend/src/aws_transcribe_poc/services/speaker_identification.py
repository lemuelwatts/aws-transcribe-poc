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

import logging
from pathlib import Path

import numpy as np
import torchaudio
from scipy.spatial.distance import cosine

from speechbrain.inference.speaker import SpeakerRecognition

logger = logging.getLogger(__name__)


class SpeakerIdentification:
    def __init__(self):
        model_path = (
            Path(__file__).parent.parent
            / "models"
            / "speechbrain-spkrec-ecapa-voxceleb"
        )

        if model_path.exists() and (model_path / "hyperparams.yaml").exists():
            logger.info(f"Loading model from local cache: {model_path}")
            self.model = SpeakerRecognition.from_hparams(
                source=str(model_path),  # use local path as source
                savedir=str(model_path)
            )
        else:
            logger.info(f"Model not found locally, downloading to: {model_path}")
            self.model = SpeakerRecognition.from_hparams( # download from HF if not locally found
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_path), # then store it locally
            )
        self.default_threshold = 0.85

    def generate_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from an audio file."""
        audio_path_obj = Path(audio_path)

        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            signal, _ = torchaudio.load(str(audio_path_obj))

            embedding = self.model.encode_batch(signal) # returns Torch Tensor, shape [batch_size, embedding_dim]

            logger.info(f"Generated embedding for {audio_path}")
            # remove dimension of size 1 bc we are processing audio file 1 at a time
            # convert to numpy bc np.savez expects numpy arrays, not torch tensors
            # convert to cpu to store as numpy; a numpy requirement
            return embedding.squeeze().cpu().numpy()  
        
        except Exception as e:
            logger.error(f"Failed to generate embedding for {audio_path}: {e}")
            raise

    def compare_speakers(
        self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.85
    ) -> tuple[float, bool]:
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

        logger.debug(
            f"Speaker comparison: similarity={similarity:.3f}, match={is_match}"
        )
        return similarity, is_match

    def verify_speakers(self, audio_path1: str, audio_path2: str) -> tuple[float, bool]:
        """Verify if two audio files are from the same speaker.

        Returns:
            Tuple of (similarity_score, prediction)
            - similarity_score: Float indicating similarity
            - prediction: Boolean (True = same speaker, False = different speaker)
        """
        try:
            score, prediction = self.model.verify_files(audio_path1, audio_path2)
            logger.info(f"Verification score: {score}, Same speaker: {prediction}")
            return float(score), bool(prediction)
        except Exception as e:
            logger.error(f"Failed to verify speakers: {e}")
            raise

    def find_match(self, audio_path: str,
        stored_embeddings: dict[str, np.ndarray],
        threshold: float | None = None
    ) -> tuple[str | None, float]: 
        """Find the best matching speaker."""
        if threshold is None:
            threshold = self.default_threshold
        
        if not stored_embeddings:
            logger.warning("No stored embeddings provided")
            return None, 0.0
        
        try:
            input_embedding = self.generate_embedding(audio_path)
            
            best_match_name = None
            best_similarity = -1.0
            
            for speaker_name, stored_embedding in stored_embeddings.items():
                similarity = 1 - cosine(input_embedding, stored_embedding)
                
                logger.debug(f"Comparing with '{speaker_name}': similarity={similarity:.3f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_name = speaker_name
            
            if best_similarity >= threshold:
                logger.info(f"Match found: '{best_match_name}' (similarity: {best_similarity:.3f})")
                return best_match_name, best_similarity
            else:
                logger.info(f"No match above threshold {threshold}. Best was {best_similarity:.3f}")
                return None, best_similarity
                
        except Exception as e:
            logger.exception(f"Error finding match for {audio_path}: {e}")
            return None, 0.0