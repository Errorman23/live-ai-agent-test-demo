from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    # Deterministic rough estimate to avoid model/tokenizer coupling in core logic.
    return max(1, math.ceil(len(text) / 4))


def compress_prompt(prompt: str, budget_tokens: int) -> str:
    lines = [line.rstrip() for line in prompt.splitlines()]

    # Drop non-essential trace lines first.
    filtered: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            filtered.append("")
            continue
        lower = s.lower()
        if lower.startswith("trace:") or lower.startswith("[trace]"):
            continue
        filtered.append(line)

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for line in filtered:
        key = line.strip()
        if not key:
            deduped.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    collapsed = "\n".join(deduped).strip()
    if not collapsed:
        collapsed = prompt[: max(32, budget_tokens * 4)]

    if estimate_tokens(collapsed) <= budget_tokens:
        return collapsed

    # Truncate low-priority lines from the end in deterministic order.
    working = deduped[:]
    while working and estimate_tokens("\n".join(working)) > budget_tokens:
        working.pop()

    truncated = "\n".join(working).strip()
    if truncated and estimate_tokens(truncated) <= budget_tokens:
        return truncated

    # Final deterministic hard cutoff.
    max_chars = max(16, budget_tokens * 4)
    return collapsed[:max_chars]
