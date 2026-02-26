from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationMetricResult:
    bertscore_f1: float | None
    error: str | None = None


def compute_bertscore_f1(*, candidate_text: str, reference_text: str) -> TranslationMetricResult:
    candidate = (candidate_text or "").strip()
    reference = (reference_text or "").strip()
    if not candidate or not reference:
        return TranslationMetricResult(bertscore_f1=None, error="Candidate/reference text is empty")

    try:
        from bert_score import score as bert_score  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return TranslationMetricResult(
            bertscore_f1=None,
            error=f"bert-score unavailable: {exc}",
        )

    try:
        _, _, f1 = bert_score(
            [candidate],
            [reference],
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        value = float(f1[0].item())
    except Exception as exc:  # noqa: BLE001
        return TranslationMetricResult(bertscore_f1=None, error=f"BERTScore failed: {exc}")

    return TranslationMetricResult(bertscore_f1=value, error=None)
