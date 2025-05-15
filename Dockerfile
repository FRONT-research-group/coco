ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS python-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install build essentials + Prolog deps
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends -y \
    gcc g++ build-essential curl libffi-dev swi-prolog \
    && rm -rf /var/lib/apt/lists/*

# --- uv layer
FROM python-base AS uv-base

ARG UV_VERSION=0.4.10
ARG UV_LINK_MODE=copy

ENV PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_VERSION=${UV_VERSION} \
    UV_LINK_MODE=${UV_LINK_MODE}

RUN pip install uv==$UV_VERSION

# --- build environment
FROM uv-base AS build-base

ARG USER=user

ENV HOME=/home/${USER}
ENV HF_HOME=${HOME}/.cache/huggingface \
    WORKSPACE=${HOME}/app/

RUN mkdir -p ${WORKSPACE}
WORKDIR ${WORKSPACE}

COPY pyproject.toml requirements.lock ${WORKSPACE}/
RUN uv pip install --no-cache --system -r requirements.lock

# --- app source
FROM build-base AS build

ARG PACKAGE_NAME=coco
ARG PORT=8000

ENV PORT=${PORT} \
    PACKAGE_NAME=${PACKAGE_NAME}

# Copy app source code
COPY ./src/${PACKAGE_NAME} ${WORKSPACE}/${PACKAGE_NAME}

# Expose model path via ENV
ENV MODEL_DIR=/models

# Port exposure for FastAPI
EXPOSE ${PORT}

# --- runtime user
ARG USER
ARG UID=10001

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "${HOME}" \
    --no-create-home \
    --uid "${UID}" \
    ${USER} && \
    chown -R ${USER} ${HOME}

USER ${USER}

# Start API
CMD ["uvicorn", "coco.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
