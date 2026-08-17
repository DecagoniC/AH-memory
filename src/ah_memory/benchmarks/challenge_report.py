"""Auditable, secret-safe challenge benchmark report output."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_SECRET_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "credentials",
        "password",
        "refresh_token",
        "token",
    }
)


def write_challenge_report(
    summary: Mapping[str, Any],
    logs: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    root: str | Path = "results/challenge",
    run_id: str | None = None,
) -> Path:
    """Write summary JSON and named JSONL audit streams."""
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(root) / identifier
    output.mkdir(parents=True, exist_ok=False)
    clean_summary = _redact(dict(summary))
    (output / "summary.json").write_text(
        json.dumps(clean_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for name, rows in logs.items():
        safe_name = _safe_filename(name)
        payload = "\n".join(
            json.dumps(_redact(dict(row)), ensure_ascii=False, sort_keys=True)
            for row in rows
        )
        if payload:
            payload += "\n"
        (output / f"{safe_name}.jsonl").write_text(payload, encoding="utf-8")
    return output


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).casefold() in _SECRET_KEYS
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _safe_filename(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    ).strip("_")
    if not cleaned:
        raise ValueError("log name must contain a filename-safe character")
    return cleaned
