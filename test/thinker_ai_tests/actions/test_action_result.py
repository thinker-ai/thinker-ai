import unittest
from pathlib import Path

from thinker_ai.actions import ActionResult


def get_project_root() -> Path:
    """逐级向上寻找项目根目录"""
    current_path = Path.cwd()
    while True:
        if (current_path / '.git').exists() or \
                (current_path / '.project_root').exists() or \
                (current_path / '.gitignore').exists():
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            raise Exception("Project root not found.")
        current_path = parent_path


class TestActionResult(unittest.TestCase):

    def test_plain_text(self):
        # 测试纯文本的解析
        text = "## Title\nThis is a simple text block."
        mapping = {"Title": (str, ...)}
        action_result = ActionResult.loads(text, mapping)
        self.assertEqual(action_result.instruct_content.Title, "This is a simple text block.")

    def test_list_parsing(self):
        # 测试列表的解析
        text = self.load_test_file("system_requirement.md")
        mapping = {"List": (list, ...)}
        action_result = ActionResult.loads(text, mapping)
        self.assertEqual(action_result.instruct_content.List, ['item1', 'item2'])

    def test_dict_parsing(self):
        # 测试字典的解析
        text = "## Dictionary\nkey1: value1\nkey2: value2"
        mapping = {"Dictionary": (dict, ...)}
        action_result = ActionResult.loads(text, mapping)
        self.assertEqual(action_result.instruct_content.Dictionary, {'key1': 'value1', 'key2': 'value2'})

    def load_test_file(self, file_name: str) -> str:
        if file_name.startswith('/'):
            file_name = file_name[1:]  # 否则会误判为根路径
        file_dir = get_project_root() / "test/thinker_ai_tests/actions"
        return self.load_file(file_dir, file_name)

    def load_file(self, file_dir, file_name):
        file = Path(file_dir) / file_name
        with open(file, 'r') as file:
            content = file.read()
        return content


# 运行测试
if __name__ == '__main__':
    unittest.main()
