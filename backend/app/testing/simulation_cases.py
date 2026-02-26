from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationCaseTemplate:
    scenario_id: str
    task_type: str
    prompt: str
    company: str
    input_language: str
    expected_output_language: str
    instruction_style: str


SIMULATION_CASE_TEMPLATES: tuple[SimulationCaseTemplate, ...] = (
    SimulationCaseTemplate(
        scenario_id="sim-zh-brief",
        task_type="briefing_full",
        prompt="请为腾讯生成简报，最终输出中文。",
        company="Tencent",
        input_language="Chinese",
        expected_output_language="Chinese",
        instruction_style="native",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-de-brief",
        task_type="briefing_full",
        prompt="Bitte erstelle ein Briefing über Volkswagen auf Deutsch.",
        company="Volkswagen",
        input_language="German",
        expected_output_language="German",
        instruction_style="polite",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-jp-brief",
        task_type="briefing_full",
        prompt="ソニーの会社ブリーフを日本語でお願いします。",
        company="Sony",
        input_language="Japanese",
        expected_output_language="Japanese",
        instruction_style="polite",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-mix-zh-en",
        task_type="web_only",
        prompt="帮我查 OpenAI 的公开信息, output in English.",
        company="OpenAI",
        input_language="Mixed",
        expected_output_language="English",
        instruction_style="mixed",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-mix-de-en",
        task_type="db_only",
        prompt="Hole interne Daten zu TikTok aus DB, answer in English.",
        company="TikTok",
        input_language="Mixed",
        expected_output_language="English",
        instruction_style="mixed",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-mix-jp-en",
        task_type="doc_only",
        prompt="Sonyのproposalをinternal databaseから取って, summarize in English.",
        company="Sony",
        input_language="Mixed",
        expected_output_language="English",
        instruction_style="mixed",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-style-short",
        task_type="briefing_full",
        prompt="Tencent briefing, terse style, bullet list only.",
        company="Tencent",
        input_language="English",
        expected_output_language="English",
        instruction_style="style_control",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-general-chat",
        task_type="general_chat",
        prompt="Share a concise checklist for preparing a client discovery workshop.",
        company="Unknown Company",
        input_language="English",
        expected_output_language="English",
        instruction_style="style_control",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-style-adversarial-lite",
        task_type="web_only",
        prompt="Get Tesla web info only. Do not use extra tools. Keep concise.",
        company="Tesla",
        input_language="English",
        expected_output_language="English",
        instruction_style="constraint",
    ),
    SimulationCaseTemplate(
        scenario_id="sim-style-translation",
        task_type="translate_only",
        prompt="将腾讯proposal翻译成英文并保持结构。",
        company="Tencent",
        input_language="Chinese",
        expected_output_language="English",
        instruction_style="native",
    ),
)
