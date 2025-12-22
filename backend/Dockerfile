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

# Use a Python image with uv 0.5 pre-installed
FROM ghcr.io/astral-sh/uv:0.5-python3.12-bookworm-slim

ARG APP_USERNAME=appuser
ARG USER_UID=10001
ARG USER_GID=$USER_UID

RUN true \
    # Create a non-root user for the app
    && addgroup --system --gid ${USER_GID} ${APP_USERNAME} \
    && adduser --system --uid ${USER_UID} --gid ${USER_GID} --home /home/${APP_USERNAME} ${APP_USERNAME}

# If you need to install something into the container via apt-get
# do so after the "upgrade" command and before the cleanup section
# in the run command below.
RUN true \
    # Update system packages to pull in any security updates.
    && apt-get update \
    && apt-get upgrade -y \
    # Clean up package lists now that we've upgraded.
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/lists/*

# Install the project into `/app`
WORKDIR /app
RUN chown -R ${APP_USERNAME}:${APP_USERNAME} /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --python 3.12

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY --chown=${APP_USERNAME}:${APP_USERNAME} . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --python 3.12

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Application specific env settings
# This is left here for reference, but can be deleted if you
# don't need to set any env variables from the Dockerfile itself.
# ENV \
#     # This is just an example ENV TODO: replace with project specific vars
#     # Set cache dir to a writeable dir.
#     MY_CACHE="/app/my_cache/"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the FastAPI application by default
# Uses `--host 0.0.0.0` to allow access from outside the container
# Uses `--port 80` to run the fastapi server on port 80 inside the container.
CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "80", "src/aws_transcribe_poc/webapp.py"]

USER ${APP_USERNAME}
