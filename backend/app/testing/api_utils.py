from __future__ import annotations

from typing import Any

from app.schemas import TestHistoryRow


def parse_terms(value: list[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple([str(v).strip() for v in value if str(v).strip()])
    text = str(value).strip()
    if not text:
        return ()
    return tuple([item.strip() for item in text.split("||") if item.strip()])


def parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return default


def trace_url(host: str | None, trace_id: str | None, project_id: str | None = None) -> str | None:
    _ = project_id
    if not host or not trace_id:
        return None
    return f"{host.rstrip('/')}/trace/{trace_id}"


def history_rows(app: Any, *, limit: int = 50, status: str | None = None) -> list[TestHistoryRow]:
    rows: list[TestHistoryRow] = []
    ids = list(app.state.test_history_order)[-limit:]
    for test_id in reversed(ids):
        item = app.state.test_batch_registry.get(test_id)
        if not item:
            continue
        if status and str(item.get("status")) != status:
            continue
        rows.append(
            TestHistoryRow(
                test_id=test_id,
                suite=str(item.get("suite", "unknown")),
                test_domain=str(item.get("test_domain")) if item.get("test_domain") else None,
                status=str(item.get("status", "running")),  # type: ignore[arg-type]
                started_at=str(item.get("started_at", "")),
                ended_at=str(item.get("ended_at")) if item.get("ended_at") else None,
                total_cases=int(item.get("total_cases", 0)),
                completed_cases=int(item.get("completed_cases", 0)),
                pass_rate=float(item.get("pass_rate")) if item.get("pass_rate") is not None else None,
            )
        )
    return rows
