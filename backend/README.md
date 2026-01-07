# Backend

## How to Run Locally
Build module

```bash
mvnd clean install
```

Download the data

```bash
uv run aws_transcribe_poc download
```

Train the model

```bash
uv run aws_transcribe_poc train
```

Run the webapp

```bash
uv run aws_transcribe_poc serve
```

## How to Run in Docker Container
Build the module and the docker image

```bash
mvnd clean install
```

Run containerized service

```bash
docker compose up
```


## API Endpoints

### Health Check

```bash
curl http://localhost:8000/
```

### Upload and Transcribe (Single File)

Upload a local file, convert it to WAV, upload to S3, and transcribe:

```bash
# Basic usage
curl -X POST "http://localhost:8000/upload_and_transcribe" \
  -F "file=@/path/to/meeting.mp4"

# With metrics saved to S3
curl -X POST "http://localhost:8000/upload_and_transcribe?save_metrics=true" \
  -F "file=@/path/to/meeting.mp4"
```

### Upload and Transcribe (Batch)

Upload multiple files at once:

```bash
curl -X POST "http://localhost:8000/upload_and_transcribe_batch" \
  -F "files=@/path/to/meeting1.mp4" \
  -F "files=@/path/to/meeting2.mp4" \
  -F "files=@/path/to/meeting3.mp4"
```

### Transcribe Files Already in S3

Transcribe files that are already uploaded to S3:

```bash
# Single file
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"s3_uris": ["s3://my-bucket/audio/meeting.wav"]}'

# Multiple files with metrics
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_uris": [
      "s3://my-bucket/audio/meeting1.wav",
      "s3://my-bucket/audio/meeting2.wav"
    ],
    "save_metrics": true
  }'
```

### Process Audio Only (No Transcription)

Convert and upload to S3 without transcribing:

```bash
curl -X POST "http://localhost:8000/audio/process" \
  -F "file=@/path/to/meeting.mp4"
```

## Transcription Output Structure

When a file is transcribed, AWS Transcribe generates a JSON file with the following structure:

```json
{
  "jobName": "transcribe-meeting_AUDIO_ONLY-20251219143052",
  "accountId": "123456789012",
  "status": "COMPLETED",
  "results": {
    "transcripts": [
      {
        "transcript": "Full transcript text as a single string..."
      }
    ],
    "speaker_labels": {
      "channel_label": "ch_0",
      "speakers": 3,
      "segments": [
        {
          "start_time": "0.0",
          "end_time": "5.24",
          "speaker_label": "spk_0",
          "items": [
            { "start_time": "0.009", "end_time": "0.26", "speaker_label": "spk_0" },
            { "start_time": "0.27", "end_time": "0.65", "speaker_label": "spk_0" }
          ]
        }
      ]
    },
    "items": [
      {
        "start_time": "0.009",
        "end_time": "0.26",
        "alternatives": [{ "confidence": "0.9984", "content": "Hello" }],
        "type": "pronunciation",
        "speaker_label": "spk_0"
      },
      {
        "alternatives": [{ "confidence": "0.0", "content": "," }],
        "type": "punctuation"
      }
    ]
  }
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `results.transcripts[0].transcript` | The full transcript as a single string |
| `results.speaker_labels.speakers` | Number of unique speakers detected |
| `results.speaker_labels.segments` | Time-based segments grouped by speaker |
| `results.items` | Individual words with timestamps, confidence scores, and speaker labels |

### Item Types

- **`pronunciation`**: Spoken words with `start_time`, `end_time`, `confidence`, and `speaker_label`
- **`punctuation`**: Inferred punctuation (commas, periods, etc.) — no timestamps

### Output Files

For an input file `meeting.mp4`, the service generates:

| File | Location | Description |
|------|----------|-------------|
| Transcription | `s3://{bucket}/output/meeting_transcription.json` | Full AWS Transcribe output |
| Summary | `s3://{bucket}/output/meeting_summary.txt` | AI-generated meeting summary with action items |
| Metrics (optional) | `s3://{bucket}/output/metrics/meeting_metrics_{timestamp}.json` | Processing time metrics |

## aiSSEMBLE Open Inference Protocol FastAPI Implementation

This project uses [aiSSEMBLE Open Inference Protocol](https://github.com/boozallen/aissemble-open-inference-protocol) to ensure that all FastAPI routes conform to the [Open Inference Protocol](https://github.com/kserve/open-inference-protocol). For configuration and usage details, such as implementing a custom handler or setting up authorization, refer to the [aiSSEMBLE Open Inference Protocol FastAPI documentation](https://github.com/boozallen/aissemble-open-inference-protocol/blob/dev/aissemble-open-inference-protocol-fastapi/README.md).
