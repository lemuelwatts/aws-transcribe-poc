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
"""
process test videos/audio files through endpoints: `audio/process` and `/transcribe` collected end-2-end metrics
including speeds for the WAV conversion with FFMPEG, S3 upload time, and transcription time
"""
# imports
import json
import os
import requests
import sys
import time

from typing import Any
from datetime import datetime
from pathlib import Path

# configuration set up paths for api url, video directory and result/summary destination files
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
TEST_VIDEO_DIR = Path(__file__).parent / "test_files"
RESULTS_FILE = Path(__file__).parent / "baseline_results.jsonl"

def process_file(file_path: Path) -> dict[str, Any]:
    """Process single file through audio processing endpoint"""
    with open(file_path, 'rb') as f:
        files = {"file": (file_path.name, f)} # pass filename, object for files param
        response = requests.post(f"{API_BASE_URL}/audio/process", files=files, timeout=1800)
        response.raise_for_status() # raise so we don't have to check manually
        return response.json() # parse JSON string into python dictionary 

def transcribe_file(s3_uri: str)-> dict[str, Any]:
    """Transcribe file using transcription endpoint"""
    response = requests.post(
        f"{API_BASE_URL}/transcribe",
        json={"s3_uris": [s3_uri]},
        timeout=3600
    )
    response.raise_for_status()
    return response.json()

def run_pipeline(file_path: str)-> dict[str, Any]:
    """Run full pipeline for isngle file"""
    # start clock
    start = time.time()
    # get file information

    # process file
    audio_result = process_file(file_path)
    s3_uri = audio_result["s3_uri"]

    # transcribe file
    transcription = transcribe_file(s3_uri)

    # end clock
    total_time = time.time() - start

    audio_metrics = audio_result['metrics'].copy
    audio_metrics.pop('input_metadata', None) # remove the input_metadata to clean up long metrics

    # gather results
    result = {
        "timestamp": datetime.now().isoformat(),
        "filename": file_path.name,
        "s3_uri": s3_uri,
        "audio_processing": audio_result["metrics"],
        "transcription": transcription["results"][0],
        "total_pipeline_seconds": round(total_time, 3)
    }
    
    return result

def save_result(result: dict[str, Any])-> None:
    """Write result to jsonl file"""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

def generate_summary(results: list[dict[str, Any]])->None:
    """Generate summary from all results jsonl"""
    if not results:
        return
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(results)}")
    
    avg_pipeline = sum(r["total_pipeline_seconds"] for r in results) / len(results)
    avg_ffmpeg = sum(r["audio_processing"]["ffmpeg_conversion_time_seconds"] for r in results) / len(results)
    avg_s3 = sum(r["audio_processing"]["s3_upload_time_seconds"] for r in results) / len(results)
    
    print("\nAverages:")
    print(f"  Total Pipeline: {avg_pipeline:.1f}s")
    print(f"  FFmpeg: {avg_ffmpeg:.1f}s")
    print(f"  S3 Upload: {avg_s3:.1f}s")

def main():

    # print results/write file
    print("Baseline Testing")
    print(f"API: {API_BASE_URL}")
    print(f"Test Dir: {TEST_VIDEO_DIR}\n")
    
    if not TEST_VIDEO_DIR.exists():
        print(f"Error: {TEST_VIDEO_DIR} not found")
        sys.exit(1)

    extensions = {".mp4", ".avi", ".mp3", ".wav", ".webm", ".m4a"}

    # iterate over files in /test_files
    files = [f for f in TEST_VIDEO_DIR.rglob("*") if f.suffix.lower() in extensions]
    
    if not files:
        print("No test files found")
        sys.exit(1)
    
    print(f"Found {len(files)} files\n")

    results = []
    # iterate over fiels and pass through pipeline
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name}")
        
        try:
            result = run_pipeline(file_path)
            save_result(result)
            results.append(result)
            print(f"Done in {result['total_pipeline_seconds']:.1f}s\n")
        except Exception as e:
            print(f"Failed: {e}\n")
    
    # read all jsonl files and call generate summary passing the list of jsonl files
    generate_summary(results)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()