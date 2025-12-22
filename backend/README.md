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


## aiSSEMBLE Open Inference Protocol FastAPI Implementation

This project uses [aiSSEMBLE Open Inference Protocol](https://github.com/boozallen/aissemble-open-inference-protocol) to ensure that all FastAPI routes conform to the [Open Inference Protocol](https://github.com/kserve/open-inference-protocol). For configuration and usage details, such as implementing a custom handler or setting up authorization, refer to the [aiSSEMBLE Open Inference Protocol FastAPI documentation](https://github.com/boozallen/aissemble-open-inference-protocol/blob/dev/aissemble-open-inference-protocol-fastapi/README.md).
