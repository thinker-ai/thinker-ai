import json
import re
from typing import List

from thinker_ai.actions import BaseAction
from thinker_ai.agent.tools.search_engine import SearchEngine
from thinker_ai.utils.logs import logger

SYSTEM_PROMPT_FOR_SEARCH = """
Based on the question, write a list of the most effective search statements, using the following format:
'''list
[
statements1,
statements2,
statements3,
]
'''
"""

SYSTEM_PROMPT_FOR_ANSWER = """
Based on the information,please reply to the Question:{Question}
"""


class SearchAndSummarize:
    search_engine = SearchEngine()

    @classmethod
    async def run(cls,question:str)->str:
        logger.debug(question)
        queries = await cls._get_queries(question)
        results = await cls._batch_query(queries)
        PROMPT_FOR_ANSWER = SYSTEM_PROMPT_FOR_ANSWER.format(Question=question)
        answer = await BaseAction._a_generate_stream(results, PROMPT_FOR_ANSWER)
        logger.debug(answer)
        return answer
    @classmethod
    async def _batch_query(cls, queries,per_query_max_results:int=3) -> str:
        results: dict = {}
        for query in queries:
            query_result = await cls.search_engine.run(query=query,max_results=per_query_max_results) # 因为存在@overload的run，所以await会有不正确的提示。
            if query_result:
                results[query] = query_result
        return json.dumps(results, indent=4, ensure_ascii=False)
    @classmethod
    async def _get_queries(cls,question) -> List:
        statements = await BaseAction._a_generate_stream(question, SYSTEM_PROMPT_FOR_SEARCH)
        queries = cls._to_list(statements)
        return queries
    @classmethod
    def _to_list(cls, statements: str) -> List[str]:
        # 提取'''与'''之间的内容
        between_quotes_pattern = re.compile(r"'''\s*(.*?)\s*'''", re.DOTALL)
        match = between_quotes_pattern.search(statements)
        if not match:
            logger.error("Couldn't find content between triple quotes.")
            return []

        content = match.group(1)

        # 去掉所有换行符
        content_one_line = content.replace('\n', '')

        # 提取[]中的内容
        between_brackets_pattern = re.compile(r"\[(.*?)\]")
        match = between_brackets_pattern.search(content_one_line)
        if not match:
            logger.error("Couldn't find content between brackets.")
            return []

        content_within_brackets = match.group(1)

        # 根据逗号分割并去除空格
        queries = [item.strip() for item in content_within_brackets.split(",") if item.strip()]
        return queries



