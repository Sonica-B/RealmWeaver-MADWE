# GPU image: torch 2.11 + CUDA 12.8 come from the base image; the package installs on top.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/hf REALMWEAVER_REPORTS_DIR=/app/reports
COPY pyproject.toml README.md LICENSE ./
COPY realmweaver ./realmweaver
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 ".[dev]"
COPY tests ./tests
EXPOSE 8008
CMD ["realmweaver", "serve", "--host", "0.0.0.0", "--port", "8008"]
