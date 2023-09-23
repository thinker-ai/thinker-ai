from __future__ import annotations

from enum import Enum

from thinker_ai.config import configs


class SearchEngineType(Enum):
    SERPAPI_GOOGLE = "serpapi"
    SERPER_GOOGLE = "serper"
    DIRECT_GOOGLE = "google"
    CUSTOM = "custom"
    @classmethod
    def type_of(cls, type_str: str) -> Enum | None:
        return SearchEngineType._value2member_map_.get(type_str)


search_engine_type = SearchEngineType.type_of(configs.get("search_engine"))


class WebBrowserEngineType(Enum):
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    CUSTOM = "custom"
    @classmethod
    def type_of(cls, type_str: str) -> Enum | None:
        try:
            return WebBrowserEngineType._value2member_map_.get(type_str)
        except KeyError:
            raise ValueError(f"'{type_str}' is not a valid WebBrowserEngineType.")


web_browser_engine_type = WebBrowserEngineType.type_of(configs.get("web_browser_engine"))
