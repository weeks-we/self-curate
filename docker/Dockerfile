# Build on the networked workstation; transfer with docker save/load.
# Freeze built image digests and pip freeze before cluster execution.
ARG GPU_BASE=vllm/vllm-openai@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484
FROM python@sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177 AS prepare
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs/ordinary ./configs/ordinary
RUN pip install --no-cache-dir -e . huggingface_hub datasets==5.2.0 pyarrow==26.0.1 transformers==5.22.0 smolagents==1.26.0 openai==3.57.0 evalplus==0.3.1
RUN pip check && pip freeze > /opt/environment.freeze.txt
ENTRYPOINT ["python", "-m", "self_curate.experiments.ordinary"]

FROM prepare AS grader
COPY docker/install_graders.py /tmp/install_graders.py
RUN python /tmp/install_graders.py
# Official Olympiad SymPy needs ANTLR 4.11; math-verify uses a separate stack.
RUN python -m venv /opt/olympiad && /opt/olympiad/bin/pip install sympy==1.13.3 antlr4-python3-runtime==4.11.1
ENV NLTK_DATA=/opt/nltk_data PYTHAINLP_DATA=/opt/pythainlp_data PYTHAINLP_READ_ONLY=1 PYTHAINLP_OFFLINE=1 SELF_CURATE_GRADER_SANDBOX=1 PYTHONPATH=/app/src
RUN pip freeze > /opt/environment.freeze.txt
ENTRYPOINT []

FROM ${GPU_BASE} AS gpu
RUN apt-get update && apt-get install -y --no-install-recommends docker.io git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs/ordinary ./configs/ordinary
COPY docker/ordinary-runtime.txt /tmp/ordinary-runtime.txt
RUN python -m pip install --no-cache-dir -r /tmp/ordinary-runtime.txt -e . && python -m pip check && python -m pip freeze > /opt/environment.freeze.txt
# This CUDA extension must be built/downloaded here, never on offline compute.
RUN python -m pip install --no-build-isolation causal-conv1d==1.7.0 && python -m pip check && python -m pip freeze > /opt/environment.freeze.txt
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["python", "-m", "self_curate.experiments.ordinary"]
