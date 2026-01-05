# How to test
This test folder establishes baseline running through the files included in /backend/tests/performance/test_files

### Steps
1. ensure server is running
    - `cd backend`
    - `uv run aissemble_lite_ffmpeg serve`
2. run test script
    - `cd backend/tests/performance`
    - `python baseline_test.py`

## Test flow
The script will:
- process each file through the `/audio/process` endpoint (FFMPEG + S3 Upload)
- send S3 URI to `/transcribe` endpoint (AWS)
- measure time for each stage
- save detailed results in `baseline_results.jsonl`
- print summary to console