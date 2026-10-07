# GPU image: torch 2.11 + CUDA 12.8 come from the base image; dependencies install once, the package on top.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1 HF_HOME=/hf REALMWEAVER_REPORTS_DIR=/app/reports
COPY pyproject.toml README.md LICENSE ./
# Resolve and install dependencies from a stub package so this layer is cached across code changes.
RUN mkdir -p realmweaver && touch realmweaver/__init__.py \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu128 ".[dev]" \
    && pip uninstall -y realmweaver && rm -rf realmweaver
COPY realmweaver ./realmweaver
RUN pip install --no-deps .
COPY tests ./tests
COPY unity ./unity
EXPOSE 8008
CMD ["realmweaver", "serve", "--host", "0.0.0.0", "--port", "8008"]
