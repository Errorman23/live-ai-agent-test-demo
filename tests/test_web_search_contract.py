from __future__ import annotations

from app.config import Settings
from app.internal_db.repository import InternalDBRepository
from app.tools.real_tools import RealToolbox


def test_web_search_has_no_synthetic_fallback(monkeypatch, tmp_path):
    observed_queries: list[str] = []

    def fake_tavily_search(self, query: str):  # noqa: ANN001
        observed_queries.append(query)
        return {"results": [], "answer": ""}

    monkeypatch.setattr("app.tools.real_tools.RealToolbox._tavily_search", fake_tavily_search)

    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    repository = InternalDBRepository(settings.internal_db_path)
    toolbox = RealToolbox(settings=settings, repository=repository)

    payload = toolbox.search_public_web("OpenAI").data
    assert payload["search_success"] is False
    assert payload["result_count"] == 0
    assert len(payload["query_attempts"]) == 3
    assert payload["query_attempts"][0]["query"] == "OpenAI products partnerships latest news"
    assert payload["query_attempts"][1]["query"] == "OpenAI official site products"
    assert payload["query_attempts"][2]["query"] == "OpenAI wikipedia partnerships"
    assert observed_queries == [
        "OpenAI products partnerships latest news",
        "OpenAI official site products",
        "OpenAI wikipedia partnerships",
    ]
    serialized_results = str(payload["results"]).lower()
    assert "example.com" not in serialized_results


def test_web_search_deduplicates_results(monkeypatch, tmp_path):
    responses: dict[str, list[dict[str, str]]] = {
        "OpenAI products partnerships latest news": [
            {"title": "OpenAI", "content": "Partnership update", "url": "https://openai.com/news"},
            {"title": "OpenAI duplicate", "content": "Partnership update", "url": "https://openai.com/news"},
            {"title": "OpenAI wiki", "content": "Company profile", "url": "https://en.wikipedia.org/wiki/OpenAI"},
        ]
    }

    def fake_tavily_search(self, query: str):  # noqa: ANN001
        rows = responses.get(query, [])
        return {"results": rows, "answer": ""}

    monkeypatch.setattr("app.tools.real_tools.RealToolbox._tavily_search", fake_tavily_search)

    settings = Settings(
        TOGETHER_API_KEY="test",
        REQUIRE_POSTGRES_CHECKPOINTER=False,
        INTERNAL_DB_PATH=str(tmp_path / "internal.db"),
    )
    repository = InternalDBRepository(settings.internal_db_path)
    toolbox = RealToolbox(settings=settings, repository=repository)

    payload = toolbox.search_public_web("OpenAI").data
    assert payload["search_success"] is True
    assert payload["result_count"] == 2
    assert len(payload["results"]) == 2
    assert len(payload["query_attempts"]) == 3
    assert payload["source_links"] == [
        "https://openai.com/news",
        "https://en.wikipedia.org/wiki/OpenAI",
    ]
