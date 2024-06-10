from typing import Optional, Callable

from pydantic import Field

from thinker_ai.agent.tools import SearchEngineType
from thinker_ai.utils.yaml_model import YamlModel


class SearchConfig(YamlModel):
    """Config for Search"""

    api_type: SearchEngineType = SearchEngineType.DUCK_DUCK_GO
    api_key: str = ""
    cse_id: str = ""  # for google
    search_func: Optional[Callable] = None
    params: dict = Field(
        default_factory=lambda: {
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
    )
