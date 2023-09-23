import json

import asynctest  # 导入asynctest
from unittest.mock import patch, AsyncMock

from thinker_ai.action.action import Action
from thinker_ai.tools.search_and_summarize import SearchAndSummarize
from thinker_ai.tools.search_engine import SearchEngine


class TestSearchAndSummarize(asynctest.TestCase):  # 改为继承asynctest.TestCase

    def setUp(self):
        self.question = "What is the capital of France?"

    @patch.object(SearchAndSummarize, '_get_queries', new_callable=AsyncMock)
    @patch.object(SearchAndSummarize, '_batch_query', new_callable=AsyncMock)
    @patch.object(Action, '_a_generate_stream', new_callable=AsyncMock)
    async def test_run(self, mock_a_generate, mock_batch_query, mock_get_queries):
        # 定义预期的问题和答案
        question = "What is the capital of France?"
        expected_answer = "The capital of France is Paris."

        # 设置模拟方法/函数的返回值
        mock_get_queries.return_value = ["query1", "query2"]
        mock_batch_query.return_value = '{"query1": "result1", "query2": "result2"}'
        mock_a_generate.return_value = expected_answer

        # 调用待测试的方法并捕获返回值
        result = await SearchAndSummarize.run(question)

        # 断言方法返回了预期的答案
        self.assertEqual(result, expected_answer)

    @patch.object(Action, '_a_generate_stream', new_callable=AsyncMock)
    async def test_get_queries(self, mock_generate):
        mock_generate.return_value = """
        '''list
        [
        statements1,
        statements2,
        statements3,
        ]
        '''
        """
        expected_queries = ["statements1", "statements2", "statements3"]
        result = await SearchAndSummarize._get_queries(self.question)
        self.assertEqual(result, expected_queries)


    @patch.object(SearchEngine, 'run', new_callable=AsyncMock)
    async def test_batch_query(self, mock_search_engine):
        mock_search_engine.return_value = "some results"
        queries = ["statements1"]
        result = await SearchAndSummarize._batch_query(queries)
        self.assertEqual(json.loads(result), {"statements1": "some results"})

    def test_to_list(self):
        statements = """
        '''list
        [
        statements1,
        statements2,
        statements3,
        ]
        '''
        """
        expected = ["statements1", "statements2", "statements3"]
        result = SearchAndSummarize._to_list(statements)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    asynctest.main()  # 使用asynctest的main方法

