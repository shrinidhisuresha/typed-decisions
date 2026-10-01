# typed-decisions, CPU flavour. Build:   docker build -t typed-decisions:cpu .
# Memory: the cpu preset loads Qwen3.5-0.8B in float32 (~3.5 GB peak). On a small Docker
# VM, append `typed-decisions serve --preset cpu --host 0.0.0.0 --dtype bfloat16` (~1.6 GB;
# measured slightly less accurate at 0.8B).
# Run (keys strongly recommended once the port is reachable from other machines):
#   docker run --rm -p 127.0.0.1:8080:8080 -e TD_API_KEYS=sk-change-me \
#       -v ~/.cache/huggingface:/home/app/.cache/huggingface typed-decisions:cpu
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY typed_decisions ./typed_decisions
RUN pip install '.[cpu]' && rm -rf /src

RUN useradd --create-home app
USER app
WORKDIR /home/app
EXPOSE 8080
# Inside a container the server must listen on all interfaces; publish the port on
# 127.0.0.1 (as above) unless you mean to expose it, and set TD_API_KEYS if you do.
CMD ["typed-decisions", "serve", "--preset", "cpu", "--host", "0.0.0.0", "--port", "8080"]
