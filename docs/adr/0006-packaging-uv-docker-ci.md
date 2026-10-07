# ADR-0006: One flat package, uv, Docker GPU image, CPU-only CI

Date: 2026-10-06. Status: accepted.

## Context
`setup.py` installed top-level `models`/`agents` while code imported `src.…`; CI pointed at paths that did not exist; the setup script installed CPU torch on Windows.

## Decision
Package `realmweaver` at repo root, `pyproject.toml` managed by uv (Python ≥3.11, torch from the cu128 index via `[tool.uv.sources]`, extra `gpu`), ruff for lint/format, pytest with a `gpu` marker. `Dockerfile` from `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, `compose.yaml` with `gpus: all` and mounted HF cache. CI runs ruff + CPU tests on Linux.

## Consequences
`uv sync` is the only setup command; the old `src/`, `scripts/`, `configs/`, `setup.py`, `requirements.txt` and batch file are removed; `data/processed` leaves git.

## Amendment (2026-10-06): GPU dependencies are core, there is no `gpu` extra
torch, torchvision, diffusers, peft and accelerate are plain dependencies because both targets that run the generator already provide CUDA torch: the Docker base image ships torch 2.11 + CUDA 12.8, and the laptop resolves the cu128 wheel through `[tool.uv.sources]`; Linux CI resolves the CPU wheel from the same table. An extra would only add a second way to get the same packages. The Docker image installs with pip on top of the base image's torch; `uv sync` stays the setup command everywhere else.
