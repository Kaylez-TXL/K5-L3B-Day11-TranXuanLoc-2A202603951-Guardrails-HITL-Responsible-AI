"""
Checkpoint 3 — Audit log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.security_boundary import normalize_for_security
from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[str, float, str]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Keep a redacted input and its start time until the request finishes."""
        self._open[(user_id, request_id)] = (
            utc_now_iso(),
            time.monotonic(),
            content_filter(normalize_for_security(text))["redacted"],
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store the final decision, response and elapsed time."""
        started_at, started, input_text = self._open.pop(
            (user_id, request_id), (None, None, None)
        )
        self.logs.append({
            "user_id": user_id,
            "request_id": request_id,
            "input": input_text,
            "output": content_filter(normalize_for_security(text))["redacted"],
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "latency_ms": round((time.monotonic() - started) * 1000, 3)
            if started is not None else None,
            "blocked": blocked,
            "layer": layer,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
