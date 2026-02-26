from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from app.eval import scenarios as eval_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_synthetic_companies.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("synthetic_generator", SCRIPT_PATH)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SYNTHETIC_GENERATOR = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SYNTHETIC_GENERATOR)


def _run_generator(
    *,
    db_path: Path,
    profiles_json: Path,
    manifest_json: Path,
    seed: int,
) -> dict[str, object]:
    SYNTHETIC_GENERATOR.build_dataset(
        db_path=str(db_path),
        seed=seed,
        overwrite=True,
        emit_profiles_json=str(profiles_json),
        emit_manifest_json=str(manifest_json),
    )
    return {
        "profiles": json.loads(profiles_json.read_text(encoding="utf-8")),
        "manifest": json.loads(manifest_json.read_text(encoding="utf-8")),
    }


def test_generator_emits_10_profiles_and_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    profiles_json = tmp_path / "profiles.json"
    manifest_json = tmp_path / "manifest.json"

    payload = _run_generator(
        db_path=db_path,
        profiles_json=profiles_json,
        manifest_json=manifest_json,
        seed=7,
    )
    profiles_payload = payload["profiles"]
    manifest_payload = payload["manifest"]

    assert isinstance(profiles_payload, dict)
    assert profiles_payload["profile_count"] == 10
    profiles = profiles_payload["profiles"]
    assert isinstance(profiles, list)
    assert len(profiles) == 10

    risk_categories = {str(profile["risk_category"]) for profile in profiles}
    assert {"low", "medium", "high"}.issubset(risk_categories)

    assert isinstance(manifest_payload, dict)
    assert manifest_payload["profile_count"] == 10
    assert "risk_counts" in manifest_payload
    assert "preferred_output_language_counts" in manifest_payload


def test_generator_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    profiles_a = SYNTHETIC_GENERATOR.generate_profile_records(seed=11)
    profiles_b = SYNTHETIC_GENERATOR.generate_profile_records(seed=11)
    snapshot_a = [
        (
            entry["company_name"],
            entry["risk_category"],
            entry["sensitive_fields"]["budget_usd"],
            entry["document_languages"]["proposal"],
            entry["document_languages"]["quotation"],
        )
        for entry in profiles_a
    ]
    snapshot_b = [
        (
            entry["company_name"],
            entry["risk_category"],
            entry["sensitive_fields"]["budget_usd"],
            entry["document_languages"]["proposal"],
            entry["document_languages"]["quotation"],
        )
        for entry in profiles_b
    ]
    assert snapshot_a == snapshot_b


def test_eval_scenarios_cover_all_generated_companies(tmp_path: Path, monkeypatch) -> None:
    payload = _run_generator(
        db_path=tmp_path / "scenario.db",
        profiles_json=tmp_path / "profiles.json",
        manifest_json=tmp_path / "manifest.json",
        seed=7,
    )
    profiles = payload["profiles"]["profiles"]

    from app.data.synthetic_dataset import load_synthetic_profiles

    typed_profiles = load_synthetic_profiles(profiles_path=tmp_path / "profiles.json")
    monkeypatch.setattr(eval_scenarios, "load_synthetic_profiles", lambda: typed_profiles)

    full = eval_scenarios.build_scenarios_for_suite("full")
    covered = {scenario.company_name for scenario in full if scenario.company_name}
    expected = {profile.company_name for profile in typed_profiles}
    assert expected.issubset(covered)

    leakage = eval_scenarios.build_scenarios_for_suite("leakage")
    assert any(scenario.risk_category == "high" for scenario in leakage)
