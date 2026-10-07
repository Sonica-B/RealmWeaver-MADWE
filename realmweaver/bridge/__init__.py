"""Bridge: the FastAPI service Unity and the operator page talk to (ADR-0003).

`create_app()` builds an app; the package attribute `app` is built on first access, so `uvicorn realmweaver.bridge:app`
serves without constructing a world (or loading a diffusion model) at import time.
"""

from __future__ import annotations

from fastapi import FastAPI

from realmweaver.bridge.app import create_app

# Importing the routes module binds the package attribute `app` to that module; unbind it so `realmweaver.bridge:app`
# resolves through `__getattr__` below to the FastAPI instance. The module itself stays importable by its full name.
del app  # noqa: F821

__all__ = ["app", "create_app"]

_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
