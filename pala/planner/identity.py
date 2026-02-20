from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def load_identity_text(path: str | None, fallback: str) -> str:
    """Load planner identity text from disk, with fallback to config-provided text."""
    fallback_text = str(fallback or "").strip()
    identity_path = "" if path is None else str(path).strip()
    if not identity_path:
        return fallback_text

    try:
        with open(identity_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as exc:
        logger.warning("identity file unavailable path=%s detail=%s; using fallback identity", identity_path, exc)
        return fallback_text

    text = raw.strip()
    if not text:
        logger.warning("identity file empty path=%s; using fallback identity", identity_path)
        return fallback_text

    logger.info("identity loaded path=%s chars=%d", os.path.abspath(identity_path), len(text))
    return text
