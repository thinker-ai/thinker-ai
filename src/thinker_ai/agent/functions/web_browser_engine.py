#!/usr/bin/env python

from __future__ import annotations

import importlib
from typing import Any, Callable, Coroutine, Literal, overload

from thinker_ai.agent.functions import web_browser_engine_type, WebBrowserEngineType
from thinker_ai.utils.html_parser import WebPage


class WebBrowserEngine:
    def __init__(
        self,
        run_func: Callable[..., Coroutine[Any, Any, WebPage | list[WebPage]]] | None = None,
    ):
        if web_browser_engine_type == WebBrowserEngineType.PLAYWRIGHT:
            module = "thinker_ai.agent.functions.web_browser_engine_playwright"
            self.run_func = importlib.import_module(module).PlaywrightWrapper().run
        elif web_browser_engine_type == WebBrowserEngineType.SELENIUM:
            module = "thinker_ai.agent.functions.web_browser_engine_selenium"
            self.run_func = importlib.import_module(module).SeleniumWrapper().run
        elif web_browser_engine_type == WebBrowserEngineType.CUSTOM:
            self.run_func = run_func
        else:
            raise NotImplementedError


    @overload
    async def run(self, url: str) -> WebPage:
        ...

    @overload
    async def run(self, url: str, *urls: str) -> list[WebPage]:
        ...

    async def run(self, url: str, *urls: str) -> WebPage | list[WebPage]:
        return await self.run_func(url, *urls)


if __name__ == "__main__":
    import fire

    async def main(url: str, *urls: str, engine_type: Literal["playwright", "selenium"] = "playwright", **kwargs):
        return await WebBrowserEngine(WebBrowserEngineType(engine_type), **kwargs).run(url, *urls)

    fire.Fire(main)
