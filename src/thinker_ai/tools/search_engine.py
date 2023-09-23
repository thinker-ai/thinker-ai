from __future__ import annotations

import importlib
from typing import Callable, Coroutine, Literal, overload

from thinker_ai.tools import SearchEngineType, search_engine_type

class SearchEngine:
    """Class representing a search engine."""
    def __init__(self,
                 run_func: Callable[[str, int, bool], Coroutine[None, None, str | list[str]]] = None
                 ):
        if search_engine_type == SearchEngineType.SERPAPI_GOOGLE:
            module = "thinker_ai.tools.search_engine_serpapi"
            self.run_func = importlib.import_module(module).SerpAPIWrapper().run
        elif search_engine_type == SearchEngineType.SERPER_GOOGLE:
            module = "thinker_ai.tools.search_engine_serper"
            self.run_func = importlib.import_module(module).SerperWrapper().run
        elif search_engine_type == SearchEngineType.DIRECT_GOOGLE:
            module = "thinker_ai.tools.search_engine_googleapi"
            self.run_func = importlib.import_module(module).GoogleAPIWrapper().run
        elif search_engine_type == SearchEngineType.CUSTOM:
            self.run_func = run_func
        else:
            raise NotImplementedError()

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[True] = True,
    ) -> str:
        ...

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[False] = False,
    ) -> list[dict[str, str]]:
        ...

    async def run(self, query: str, max_results: int = 8, as_string: bool = True) -> str | list[dict[str, str]]:
        """Run a search query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return. Defaults to 8.
            as_string: Whether to return the results as a string or a list of dictionaries. Defaults to True.

        Returns:
            The search results as a string or a list of dictionaries.
        """
        return await self.run_func(query, max_results=max_results, as_string=as_string)
