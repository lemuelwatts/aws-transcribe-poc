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
import os
import time
import ffmpeg
import logging

from pathlib import Path

logger = logging.getLogger(__name__)

class FfmpegHandler():
    def __init__(self):
        logger.info('initializing ffmpeg object')

    def convert_to_wav(self, input_file) -> tuple[str, dict]:
        """This function will accept a file and convert it into a wav file and return a new file
        Args:
            input_file: Path to input file
            
        Returns:
            Tuple of (output_filename, metrics_dict)        
        """

        try:
            input_path = Path(input_file)
            output_name = f'{input_path.stem}_converted.wav'

            # collect metrics
            input_size_bytes = os.path.getsize(input_file)
            input_size_mb = input_size_bytes / (1024 * 1024)
            
            logger.info(f"Converting {input_file} ({input_size_mb:.2f} MB) to WAV format")

            convert_start = time.time()
            ffmpeg.input(input_file).output(
                output_name,
                ar=16000, # 16kHz sample rate
                ac=1, # mono channel 
                acodec='pcm_s16le' # 16-bit PCM
            ).run(overwrite_output=True)
            convert_duration = time.time() - convert_start

            output_size_bytes = os.path.getsize(output_name)
            output_size_mb = output_size_bytes / (1024 * 1024)

            metrics = {
                'ffmpeg_conversion_time_seconds': round(convert_duration, 3),
                'input_file_size_bytes': input_size_bytes,
                'input_file_size_mb': round(input_size_mb, 2),
                'output_file_size_bytes': output_size_bytes,
                'output_file_size_mb': round(output_size_mb, 2),
                'size_reduction_percent': round((1 - output_size_bytes/input_size_bytes) * 100, 2) if input_size_bytes > 0 else 0
            }
            
            logger.info(f"Conversion complete: {output_name} ({convert_duration:.2f}s, {output_size_mb:.2f} MB)")
            return output_name, metrics
        
        except Exception as e:
            logger.error(f"Ffmpeg conversion failed - {input_file}: {str(e)}")
            raise RuntimeError(f"FFmpeg conversion failed for {input_file}: {str(e)}")

    def probe_file(self, file_path: str)->dict:
        """ Use ffmpeg probe to get metadata """
        return ffmpeg.probe(file_path)