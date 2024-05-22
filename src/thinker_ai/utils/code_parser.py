import ast
import re

from thinker_ai.common.logs import logger


class CodeParser:

    @classmethod
    def parse_block(cls, block: str, text: str) -> str:
        blocks = cls.parse_blocks(text)
        for k, v in blocks.items():
            if block in k:
                return v
        return ""

    @classmethod
    def parse_blocks(cls, text: str):
        # 首先根据"##"将文本分割成不同的block
        blocks = text.split("##")

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() != "":
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = block.split("\n", 1)
                block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, block: str, text: str, lang: str = "") -> str:
        if block:
            text = cls.parse_block(block, text)
        pattern = rf'```{lang}.*?\s+(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            logger.error(f"{pattern} not match following text:")
            logger.error(text)
            raise Exception
        return code

    @classmethod
    def parse_str(cls, block: str, text: str, lang: str = ""):
        code = cls.parse_code(block, text, lang)
        code = code.split("=")[-1]
        code = code.strip().strip("'").strip("\"")
        return code

    @classmethod
    def parse_file_list(cls, block: str, text: str, lang: str = "") -> list[str]:
        # Regular expression pattern to find the works list.
        code = cls.parse_code(block, text, lang)
        # print(code1)
        pattern = r'\s*(.*=.*)?(\[.*\])'

        # Extract works list string using regex.
        match = re.search(pattern, code, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            tasks = ast.literal_eval(tasks_list_str)
        else:
            raise Exception
        return tasks