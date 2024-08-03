from __future__ import annotations

import json

import numpy as np
from overrides import overrides
from pydantic import BaseModel, field_validator
from rank_bm25 import BM25Okapi

from thinker_ai.agent.provider.llm import LLM
from thinker_ai.common.logs import logger
from thinker_ai.agent.tools import TOOL_REGISTRY
from thinker_ai.agent.tools.tool_data_type import Tool
from thinker_ai.agent.tools.tool_registry import get_register_tools
from thinker_ai.status_machine.task_desc import TaskDesc
from thinker_ai.utils.code_parser import CodeParser

TOOL_INFO_PROMPT = """
## Capabilities
- You can utilize pre-defined tools in any code lines from 'Available Tools' in the form of Python class or function.
- You can freely combine the use of any other public packages, like sklearn, numpy, pandas, etc..

## Available Tools:
Each tool is described in JSON format. When you call a tool, import the tool from its path first.
{tool_schemas}
"""
TOOL_RECOMMENDATION_PROMPT = """
## Parent Task:
    instruction:{parent_instruction}
    plan:{parent_plan}
    
## Current Task:
    {task_desc}
    
## Available Tools:
    {available_tools}

## Your To do:
Recommend up to {top_k} tools from 'Available Tools' that can help solve the 'Current Task'. 

## Tool Selection and Instructions:
- Select tools most relevant to completing the 'Current Task'.
- If you believe that no tools are suitable, indicate with an empty list.
- Only list the names of the tools, not the full schema of each tool.
- Ensure selected tools are listed in 'Available Tools'.
- Output a json list of tool names:
```json
["tool_name1", "tool_name2", ...]
```
"""


class ToolRecommender(BaseModel):
    tools: dict[str, Tool]

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, v: list[str]) -> dict[str, Tool]:
        # One can use special symbol ["<all>"] to indicate use of all registered tools
        if v == ["<all>"]:
            return TOOL_REGISTRY.get_all_tools()
        else:
            return get_register_tools(v)

    async def get_recommended_tool_info_by_default(
            self,
            parent_task_desc: TaskDesc,
            task_desc: TaskDesc,
            recall_top_k:int=20,
            top_k: int = 5
    ) -> str:
        tool_matcher = ToolMatcher(
            tools=self.tools,
            recall_top_k=recall_top_k
        )
        return await self._get_ranked_tools_info(
            tool_matcher=tool_matcher,
            parent_task_desc=parent_task_desc,
            task_desc=task_desc,
            top_k=top_k)

    async def get_recommended_tool_info_by_tag(
            self,
            parent_task_desc: TaskDesc,
            task_desc: TaskDesc,
            tool_tag: str,
            recall_top_k: int = 20,
            rank_top_k: int = 5
    ) -> str:
        tool_matcher = ToolTagMatcher(
            tools=self.tools,
            tool_tag=tool_tag,
            recall_top_k=recall_top_k
        )
        return await self._get_ranked_tools_info(
            tool_matcher=tool_matcher,
            parent_task_desc=parent_task_desc,
            task_desc=task_desc,
            top_k=rank_top_k)

    async def get_recommended_tool_info_by_bm25(
            self,
            parent_task_desc: TaskDesc,
            task_desc: TaskDesc,
            recall_top_k: int = 20,
            rank_top_k: int = 5
    ) -> str:
        tool_matcher = BM25ToolMatcher(
            tools=self.tools,
            task_desc=task_desc,
            recall_top_k=recall_top_k
        )
        return await self._get_ranked_tools_info(
            tool_matcher=tool_matcher,
            parent_task_desc=parent_task_desc,
            task_desc=task_desc,
            top_k=rank_top_k)

    async def _get_ranked_tools_info(self,
                                     tool_matcher: ToolMatcher,
                                     parent_task_desc: TaskDesc,
                                     task_desc: TaskDesc,
                                     top_k) -> str:
        recalled_tools = await tool_matcher.recall_tools()
        if not recalled_tools:
            return ""
        ranked_tools = await self.rank_tools(recalled_tools=recalled_tools,
                                             parent_task_desc=parent_task_desc,
                                             task_desc=task_desc,
                                             top_k=top_k)
        logger.info(f"Recommended tools: \n{[tool.name for tool in ranked_tools]}")
        tool_schemas = {tool.name: tool.schemas for tool in ranked_tools}
        return TOOL_INFO_PROMPT.format(tool_schemas=json.dumps(tool_schemas, indent=4,ensure_ascii=False))

    async def rank_tools(self,
                         recalled_tools: list[Tool],
                         parent_task_desc: TaskDesc,
                         task_desc: TaskDesc,
                         top_k: int = 5
                         ) -> list[Tool]:
        """
        Default rank methods for a ToolRecommender. Use LLM to rank the recalled tools based on the given requirement and topk value.
        """
        available_tools = {tool.name: tool.schemas["description"] for tool in recalled_tools}
        prompt = TOOL_RECOMMENDATION_PROMPT.format(
            parent_instruction=parent_task_desc.instruction,
            parent_plan=json.dumps(parent_task_desc.plan, indent=4,ensure_ascii=False),
            task_desc=task_desc.to_string(),
            available_tools=json.dumps(available_tools, indent=4,ensure_ascii=False),
            top_k=top_k,
        )
        rsp = await LLM().aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        ranked_tools = json.loads(rsp)
        valid_tools = get_register_tools(ranked_tools)
        return list(valid_tools.values())[:top_k]


class ToolMatcher:
    tools: dict[str, Tool]
    recall_top_k: int

    def __init__(self, tools: dict[str, Tool], recall_top_k: int = 20):
        self.tools = tools
        self.recall_top_k = recall_top_k

    async def recall_tools(self) -> list[Tool]:
        return list(self.tools.values())


class ToolTagMatcher(ToolMatcher):
    def __init__(self, tools: dict[str, Tool],
                 tool_tag: str,
                 recall_top_k: int = 20
                 ):
        super().__init__(tools, recall_top_k)
        self.tool_tag = tool_tag[:self.recall_top_k]

    @overrides
    async def recall_tools(self) -> list[Tool]:
        if not self.tool_tag:
            return list(self.tools.values())[:self.recall_top_k]
        # find tools based on exact match between task type and tool tag
        candidate_tools = TOOL_REGISTRY.get_tools_by_tag(self.tool_tag)
        candidate_tool_names = set(self.tools.keys()) & candidate_tools.keys()
        recalled_tools = [candidate_tools[tool_name] for tool_name in candidate_tool_names][:self.recall_top_k]
        logger.info(f"Recalled tools: \n{[tool.command for tool in recalled_tools]}")
        return recalled_tools


class BM25ToolMatcher(ToolMatcher):

    def __init__(self, tools: dict[str, Tool],
                 task_desc: TaskDesc,
                 recall_top_k: int
                 ):
        super().__init__(tools, recall_top_k)
        if len(self.tools) > 0:
            corpus = [f"{tool.name} {tool.tags}: {tool.schemas['description']}" for tool in self.tools.values()]
            tokenized_corpus = [self._tokenize(doc) for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.task_desc = task_desc

    def _tokenize(self, text):
        return text.split()  # FIXME: needs more sophisticated tokenization

    @overrides
    async def recall_tools(self) -> list[Tool]:
        if not self.tools:
            return []
        task_desc_tokens = self._tokenize(self.task_desc.to_string())
        doc_scores = self.bm25.get_scores(task_desc_tokens)
        top_indexes = np.argsort(doc_scores)[::-1][:self.recall_top_k]
        recalled_tools = [list(self.tools.values())[index] for index in top_indexes]

        logger.info(
            f"Recalled tools: \n{[tool.name for tool in recalled_tools]}; Scores: {[np.round(doc_scores[index], 4) for index in top_indexes]}"
        )

        return recalled_tools


class EmbeddingToolMatcher(ToolMatcher):
    def __init__(self, tools: dict[str, Tool], recall_top_k: int):
        super().__init__(tools, recall_top_k)

    @overrides
    async def recall_tools(self) -> list[Tool]:
        pass
