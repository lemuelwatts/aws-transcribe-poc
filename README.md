# aws-transcribe-poc

Someone generated the project without updating the description.

## Overview

## Requirements

aws-transcribe-poc requires the following tools to be installed on your system to build and run the project:

- [python](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [rust](https://www.rust-lang.org/tools/install)
- [hawkeye 6.0.0+](https://github.com/korandoru/hawkeye?tab=readme-ov-file#cargo-install)
- [ruff](https://docs.astral.sh/ruff/installation/) (Optional)
- [docker](https://docs.docker.com/get-docker/) or [rancher](https://rancherdesktop.io/)
- [mvnd](https://github.com/apache/maven-mvnd?tab=readme-ov-file#how-to-install-mvnd)1.x (__Note:__ currently, Homebrew will install 2.x release candidates by default.)

If you are using Rancher Desktop, you will need to add the following line to `.bashrc`, `.zshrc`, or similar:

```bash
export DOCKER_HOST=unix://$HOME/.rd/docker.sock
```

## Development Environment

### Build the project

Run `mvnd clean install` to build the project including the containers.

> [!NOTE]
> To stream logs in real time, build a single module using the `-pl` flag (e.g., `mvnd clean install -pl :backend-docker`) \
> If the build fails, rerun the module with the `-e` flag to get detailed stack traces (e.g., `mvnd clean install -pl :backend-docker -e`).

### Backend

All commands in this section should be run from the `backend` directory.

```bash
cd backend
```

Download the dataset

```bash
uv run python -m aws_transcribe_poc download
```

Train the model

```bash
uv run python -m aws_transcribe_poc train
```

Run the backend service

```bash
uv run python -m aws_transcribe_poc serve
```


## Demo Instructions

When demoing, run the containerized frontend and backend services using the following command:

```bash
docker compose up --build
```
