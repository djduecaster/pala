"""Web probe sidecar for env payload inspection and diagnostics."""

from .app import app, create_app

__all__ = ["app", "create_app"]
