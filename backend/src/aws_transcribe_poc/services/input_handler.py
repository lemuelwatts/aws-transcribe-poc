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
from pathlib import Path
import time

from .ffmpeg_handler import FfmpegHandler
from .s3_handler import S3Handler

ACCEPTABLE_TYPES = {'.mp3', '.mp4', '.wav', '.m4a', '.flac', '.avi', '.webm'}  # Use set for O(1) lookup

logger = logging.getLogger(__name__)

class InputHandler():
    def __init__(self):
        self.ffmpeg_handler = FfmpegHandler()
        self.s3_handler = S3Handler()
        self.output_dir = Path("output/metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_metrics = os.getenv('SAVE_METRICS', 'true').lower() == 'true' # make it boolean by adding comperative operator

    def _check_file_extension(self, file_path: str):
        # check if file exists, raise error if not found
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')

        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix

        if extension not in ACCEPTABLE_TYPES:
            raise ValueError(f"Unsupported file type: {extension}. Supported files: {ACCEPTABLE_TYPES}")

    def _probe_file(self, file_path: str) -> dict:
        """Extract metadata using ffprobe"""
        probe = self.ffmpeg_handler.probe_file(file_path)
        logger.info(f'probe info for {file_path} -> {probe}\n')
        format_info = probe['format']
        logger.info(f'format: {format_info}')
        streams_info = probe['streams']
        logger.info(f'streams info: {streams_info}')        
        return {
            'format': format_info,
            'streams': streams_info
        }

    def _ensure_wav(self, input_file_path):
        """This function accepts an input file and returns .wav format. Always to ensure it is standardized."""
        output, metrics = self.ffmpeg_handler.convert_to_wav(input_file_path)
        
        return output, metrics
    
    def _save_metrics(self, metrics: dict, filename: str, s3_uri:str)->None:
        """
        This function will accept metrics from the filename and its s3 uri and write
        them to the output directory
        """
        filename_stem = Path(filename).stem
        metrics_filename = f"{filename_stem}.json"
        metrics_path = self.output_dir / metrics_filename

        full_metrics = {
            'filename': filename,
            's3_uri': s3_uri,
            'metrics': metrics
        }

        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=2)

        logger.info(f'metrics saved to {metrics_path}')

    def process_input(self, input_file_path: str, original_filename: str) -> tuple[str, dict]:
        """This function ensures the file is in .wav format and gets a uri from the s3
            bucket so that we can pass a uri to the aws transcribe function.

            FFmpegHandler class is used to convert the audio as well as standardize the
            audio input for the highest quality. 
        
        """
        start_time = time.time()
        wav_filepath = None
        try:
            # check if input extension is in acceptable types
            self._check_file_extension(input_file_path)

            input_metadata = self._probe_file(file_path=input_file_path)

            wav_filepath, ffmpeg_metrics = self._ensure_wav(input_file_path)
            
            s3_uri, s3_metrics = self.s3_handler.upload_file(file_path=wav_filepath, original_filename=original_filename)

            if not s3_uri:
                raise RuntimeError(f"Failed to upload {wav_filepath} to S3")
            
            total_time = time.time() - start_time

            metrics = {
                'total_processing_time_seconds': round(total_time, 3),
                'input_metadata': input_metadata,
                **ffmpeg_metrics,
                **s3_metrics
            }

            self._save_metrics(metrics, original_filename, s3_uri)

            return s3_uri, metrics
        
        finally:
            if wav_filepath and os.path.exists(wav_filepath):
                try:
                    os.unlink(wav_filepath)
                    logger.info(f"cleaned up temp WAV file: {wav_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {wav_filepath}, {str(e)}")