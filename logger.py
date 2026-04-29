"""
Structured logging for the Mood Machine.

Writes one JSON record per line to mood_machine.log.
The console handler is WARNING-level only so the interactive loop stays clean.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


_LOG_FILE = Path(__file__).parent / "mood_machine.log"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = dict(record.msg) if isinstance(record.msg, dict) else {"message": record.getMessage()}
        payload["level"] = record.levelname
        payload["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        return json.dumps(payload)


def setup_logger() -> logging.Logger:
    """Return the mood_machine logger, creating handlers on first call."""
    logger = logging.getLogger("mood_machine")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())

    # Console only shows warnings/errors — keeps the interactive loop clean.
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(_JsonFormatter())

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_prediction(
    logger: logging.Logger,
    *,
    text: str,
    rule_pred: str,
    ml_pred: Optional[str],
    confidence: str,
    score: int,
    edge_cases: List[str],
    models_agree: Optional[bool],
) -> None:
    """Log a successful prediction at INFO level."""
    logger.info({
        "event": "prediction",
        "input": text,
        "rule_prediction": rule_pred,
        "ml_prediction": ml_pred,
        "confidence": confidence,
        "score": score,
        "edge_cases": edge_cases,
        "models_agree": models_agree,
    })


def log_edge_case_failure(
    logger: logging.Logger,
    *,
    text: str,
    flags: List[str],
) -> None:
    """Log a prediction that hit a known hard case (negation, sarcasm, etc.)."""
    logger.warning({
        "event": "edge_case_detected",
        "input": text,
        "flags": flags,
    })


def log_error(
    logger: logging.Logger,
    *,
    error_type: str,
    detail: str = "",
) -> None:
    """Log an input validation or runtime error."""
    logger.error({
        "event": "error",
        "error_type": error_type,
        "detail": detail,
    })
