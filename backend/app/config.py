from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    chainlit_host: str = Field(default="127.0.0.1", alias="CHAINLIT_HOST")
    chainlit_port: int = Field(default=8501, alias="CHAINLIT_PORT")
    testing_ui_host: str = Field(default="127.0.0.1", alias="TESTING_UI_HOST")
    testing_ui_port: int = Field(default=8502, alias="TESTING_UI_PORT")

    together_api_key: str = Field(alias="TOGETHER_API_KEY")
    together_base_url: str = Field(default="https://api.together.xyz/v1", alias="TOGETHER_BASE_URL")
    together_model: str = Field(
        default="openai/gpt-oss-20b",
        alias="TOGETHER_MODEL",
    )
    agent_reasoning_effort: str = Field(default="low", alias="AGENT_REASONING_EFFORT")

    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    llm_safety_margin_tokens: int = Field(default=512, alias="LLM_SAFETY_MARGIN_TOKENS")

    planner_max_tokens: int = Field(default=1200, alias="PLANNER_MAX_TOKENS")
    composer_max_tokens: int = Field(default=2000, alias="COMPOSER_MAX_TOKENS")
    judge_max_tokens: int = Field(default=900, alias="JUDGE_MAX_TOKENS")
    llm_judge_enabled: bool = Field(default=True, alias="LLM_JUDGE_ENABLED")
    llm_judge_model: str = Field(
        default="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        alias="LLM_JUDGE_MODEL",
    )
    llm_judge_reasoning_effort: str = Field(default="medium", alias="LLM_JUDGE_REASONING_EFFORT")

    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_host: str | None = Field(default=None, alias="LANGFUSE_HOST")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_project_id: str | None = Field(default=None, alias="LANGFUSE_PROJECT_ID")
    langfuse_postgres_uri: str | None = Field(default=None, alias="LANGFUSE_POSTGRES_URI")
    langfuse_eval_dataset_name: str = Field(
        default="agent-safety-evals",
        alias="LANGFUSE_EVAL_DATASET_NAME",
    )
    langfuse_eval_delay_seconds: int = Field(default=10, alias="LANGFUSE_EVAL_DELAY_SECONDS")
    langfuse_native_evaluator_bootstrap_enabled: bool = Field(
        default=False,
        alias="LANGFUSE_NATIVE_EVALUATOR_BOOTSTRAP_ENABLED",
    )

    # Set to false in isolated tests when no Postgres checkpointer is available.
    require_postgres_checkpointer: bool = Field(default=True, alias="REQUIRE_POSTGRES_CHECKPOINTER")
    langgraph_postgres_uri: str | None = Field(default=None, alias="LANGGRAPH_POSTGRES_URI")

    eval_default_parallelism: int = Field(default=1, alias="EVAL_DEFAULT_PARALLELISM")
    eval_repeat_count: int = Field(default=5, alias="EVAL_REPEAT_COUNT")

    internal_db_path: str = Field(default="backend/data/internal_company.db", alias="INTERNAL_DB_PATH")
    artifacts_dir: str = Field(default="backend/data/artifacts", alias="ARTIFACTS_DIR")
    templates_dir: str = Field(default="backend/app/templates", alias="TEMPLATES_DIR")
    internal_pdf_doc_type_default: str = Field(default="proposal", alias="INTERNAL_PDF_DOC_TYPE_DEFAULT")

    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    tavily_search_url: str = Field(default="https://api.tavily.com/search", alias="TAVILY_SEARCH_URL")
    tavily_search_depth: str = Field(default="advanced", alias="TAVILY_SEARCH_DEPTH")
    tavily_topic: str = Field(default="general", alias="TAVILY_TOPIC")
    tavily_max_results: int = Field(default=6, alias="TAVILY_MAX_RESULTS")

    siliconflow_api_key: str | None = Field(default=None, alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(default="https://api.siliconflow.com/v1", alias="SILICONFLOW_BASE_URL")
    siliconflow_mt_model: str = Field(default="tencent/Hunyuan-MT-7B", alias="SILICONFLOW_MT_MODEL")
    siliconflow_timeout_seconds: float = Field(default=45.0, alias="SILICONFLOW_TIMEOUT_SECONDS")

    promptfoo_enabled: bool = Field(default=True, alias="PROMPTFOO_ENABLED")
    promptfoo_command: str = Field(default="npx -y promptfoo@0.120.25", alias="PROMPTFOO_COMMAND")
    promptfoo_port: int = Field(default=15500, alias="PROMPTFOO_PORT")
    promptfoo_output_dir: str = Field(default="backend/data/artifacts/promptfoo", alias="PROMPTFOO_OUTPUT_DIR")
    promptfoo_log_path: str = Field(default=".run/promptfoo_view.log", alias="PROMPTFOO_LOG_PATH")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
