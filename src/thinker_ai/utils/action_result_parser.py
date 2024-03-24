import ast
import contextlib
import json
import re
from typing import Tuple, get_origin, Dict, Type
import yaml
from pydantic import root_validator, validator, create_model, BaseModel, field_validator, model_validator


class ActionResultParser:

    @classmethod
    def parse_blocks(cls, text: str):
        # 首先根据"##"将文本分割成不同的block
        text = text.strip()
        blocks = re.split(r'(?<!#)##(?!#)', text)

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() != "":
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = [item.strip() for item in block.split("\n", 1)]
                # LLM可能出错，在这里做一下修正
                if len(block_title) > 0 and block_title[-1] == ":":
                    block_title = block_title[:-1]
                block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_data_with_mapping(cls, data, mapping) -> BaseModel:
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception as e:
                print(f"Error: {e}")
                pass
            typing_define = mapping.get(block, None)
            if isinstance(typing_define, tuple) or isinstance(typing_define, Tuple):
                typing = typing_define[0]
            else:
                typing = typing_define
            if get_origin(typing) is list:
                # 尝试解析list
                try:
                    content = cls.parse_list(text=content)
                except Exception as e:
                    print(f"Error: {e}")
                    pass
            if get_origin(typing) is dict:
                # 尝试解析list
                try:
                    if typing==Dict[str,str]:
                        content=yaml.safe_load(content) #解决json解析器解决不了的字符换行问题，但目前只对Dict[str,str]有效
                    else:
                        content =json.loads(content) #使用json.load(),检查过于严苛，容易出现无法预期的错误
                except Exception as e:
                    print(f"Error: {e}")
                    pass
            # TODO: 多余的引号去除有风险，后期再解决
            # elif typing == str:
            #     # 尝试去除多余的引号
            #     try:
            #         content = cls.parse_str(text=content)
            #     except Exception:
            #         pass
            parsed_data[block] = content
        dynamic_model = cls._create_model_class(mapping)
        instruct_content = dynamic_model(**parsed_data)
        return instruct_content

    @classmethod
    def parse_code(cls, text: str, lang: str = "") -> str:
        pattern = rf'```{lang}.*?\s+(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise Exception
        return code

    @classmethod
    def parse_str(cls, text: str):
        text = text.split("=")[-1]
        text = text.strip().strip("'").strip("\"")
        return text

    @classmethod
    def parse_list(cls, text: str) -> list[str]:
        # Regular expression pattern to find the  list.
        pattern = r'\s*(.*=.*)?(\[.*\])'

        # Extract  list string using regex.
        match = re.search(pattern, text, re.DOTALL)
        if match:
            list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            lines= ast.literal_eval(list_str)
        else:
            lines= text.split("\n")
        filtered_list=[]
        for s in lines:
            if isinstance(s, str):
                if s.strip():
                    filtered_list.append(s.strip())
            else:
                filtered_list.append(s)
        if text.strip() == filtered_list[0]:
            raise Exception("the text can not be parsed to list")
        return filtered_list

    @staticmethod
    def parse_python_code(text: str) -> str:
        for pattern in (
                r'(.*?```python.*?\s+)?(?P<code>.*)(```.*?)',
                r'(.*?```python.*?\s+)?(?P<code>.*)',
        ):
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            code = match.group("code")
            if not code:
                continue
            with contextlib.suppress(Exception):
                ast.parse(code)
                return code
        raise ValueError("Invalid python code")

    @classmethod
    def parse_data(cls, data):
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception:
                pass

            # 尝试解析list
            try:
                content = cls.parse_list(text=content)
            except Exception:
                pass
            parsed_data[block] = content
        return parsed_data

    @classmethod
    def _create_model_class(cls, mapping: Dict)->Type[BaseModel]:
        new_class = create_model("default", **mapping)

        @field_validator('*')
        def check_name(v, field):
            if field.name not in mapping.keys():
                raise ValueError(f'Unrecognized block: {field.name}')
            return v

        @model_validator(mode='before')
        def check_missing_fields(values):
            required_fields = set(mapping.keys())
            missing_fields = required_fields - set(values.keys())
            if missing_fields:
                raise ValueError(f'Missing fields: {missing_fields}')
            return values

        new_class.__validator_check_name = classmethod(check_name)
        new_class.__root_validator_check_missing_fields = classmethod(check_missing_fields)
        return new_class

    @classmethod
    def preprocess_json_str(cls,json_str: str) -> str:
        # 替换字符串内的实际换行为 \n
        return re.sub(r'(?<=")([^"]*?)(?=")', lambda m: m.group(0).replace("\n", ""), json_str)
