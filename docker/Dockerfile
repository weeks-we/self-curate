# syntax=docker/dockerfile:1.7
# Set BASE_IMAGE to a reviewed digest in the build plan; no credentials in ARG.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git curl ca-certificates build-essential \
    docker.io docker-compose-v2 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && pip install --upgrade pip setuptools wheel
# Required full commits: resolve once in the network build environment, then
# record and reuse them. Never resolve main again in the GPU job.
ARG SWESMITH_COMMIT
ARG SWEBENCH_COMMIT
ARG HARBOR_COMMIT
RUN python -c 'import os,re; values=[os.environ.get(n, "") for n in ("SWESMITH_COMMIT","SWEBENCH_COMMIT","HARBOR_COMMIT")]; assert all(re.fullmatch("[0-9a-f]{40}", x) for x in values), "Three full harness commit IDs are required"'
WORKDIR /opt/self-curate
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e '.[server]' \
    "swesmith[validate] @ git+https://github.com/SWE-bench/SWE-smith.git@${SWESMITH_COMMIT}" \
    "swebench @ git+https://github.com/SWE-bench/SWE-bench.git@${SWEBENCH_COMMIT}" \
    "harbor @ git+https://github.com/harbor-framework/harbor.git@${HARBOR_COMMIT}"
COPY configs ./configs
COPY docs ./docs
RUN mkdir -p /opt/build-evidence && pip freeze --all > /opt/build-evidence/requirements.lock.txt \
    && python -c "import importlib.metadata as m,json; from pathlib import Path; Path('/opt/build-evidence/harnesses.json').write_text(json.dumps({n:m.distribution(n).read_text('direct_url.json') for n in ('swesmith','swebench','harbor')},indent=2))"
# Writable, persistent storage is supplied by the platform; it is not a layer
# containing credentials, experiment results or one more copy of every weight.
CMD ["python", "-m", "self_curate.server", "--help"]

# Optional large source image. No GPU and no Docker daemon are needed to build
# this target. Task images are prepared separately against the execution daemon.
FROM runtime AS with-sources
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    python -m self_curate.server --plan configs/experiments/exp001/server_bootstrap.template.json \
        --stage download --execution server
