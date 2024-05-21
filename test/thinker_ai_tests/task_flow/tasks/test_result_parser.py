import unittest
from typing import Dict, List

from thinker_ai.context import get_project_root
from thinker_ai.common.common import load_file
from thinker_ai.utils.text_parser import TextParser


class TestActionResult(unittest.TestCase):

    def test_plain_text(self):
        # 测试纯文本的解析
        text = "## Title\nThis is a simple text block."
        mapping = {"Title": (str, ...)}
        action_result = TextParser.parse_data_with_mapping(text, mapping)
        self.assertEqual(action_result.Title, "This is a simple text block.")

    def test_list_parsing(self):
        # 测试列表的解析
        text = self.load_test_file("test_list.md")
        mapping = {"test_list": (List[str], ...)}
        action_result = TextParser.parse_data_with_mapping(text, mapping)
        self.assertEqual(action_result.test_list, ['item1', 'item2'])

    def test_dict_parsing(self):
        # 测试字典的解析
        text = self.load_test_file("test_dict.md")
        mapping = {"dictionary": (Dict[str,str], ...)}
        action_result = TextParser.parse_data_with_mapping(text, mapping)
        self.assertEqual(action_result.dictionary, {'key1': 'value1', 'key2': 'value2'})

    def load_test_file(self, file_name: str) -> str:
        if file_name.startswith('/'):
            file_name = file_name[1:]  # 否则会误判为根路径
        file_dir = get_project_root() / "test/thinker_ai_tests/tasks/tasks"
        return load_file(file_dir, file_name)


# 运行测试
if __name__ == '__main__':
    unittest.main()
