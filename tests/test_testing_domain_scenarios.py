from __future__ import annotations

from app.testing.scenarios import build_domain_scenarios


def test_domain_scenarios_have_fixed_size() -> None:
    functional = build_domain_scenarios("functional")
    accuracy = build_domain_scenarios("accuracy")
    simulation = build_domain_scenarios("simulation")

    assert len(functional) == 10
    assert len(accuracy) == 10
    assert len(simulation) == 10


def test_accuracy_domain_contains_translation_cases() -> None:
    accuracy = build_domain_scenarios("accuracy")
    translate = [case for case in accuracy if case.task_type == "translate_only"]
    assert len(translate) >= 2


def test_simulation_domain_covers_multilingual_inputs() -> None:
    simulation = build_domain_scenarios("simulation")
    langs = {case.input_language for case in simulation}
    assert "Chinese" in langs
    assert "German" in langs
    assert "Japanese" in langs
    assert "Mixed" in langs


def test_simulation_translation_case_has_english_output_target() -> None:
    simulation = build_domain_scenarios("simulation")
    translation_case = next(case for case in simulation if case.scenario_id == "sim-style-translation")
    assert translation_case.input_language == "Chinese"
    assert translation_case.expected_output_language == "English"
