import re
from typing import Dict, Type

from pydantic import BaseModel, create_model, root_validator, validator

from thinker_ai.utils.logs import logger
from thinker_ai.utils.output_parser import OutputParser


class ActionOutput:
    content: str
    instruct_content: BaseModel

    def __init__(self, content: str, instruct_content: BaseModel):
        self.content = content
        self.instruct_content = instruct_content

    @classmethod
    def create_model_class(cls, class_name: str, mapping: Dict[str, Type]):
        new_class = create_model(class_name, **mapping)

        @validator('*', allow_reuse=True)
        def check_name(v, field):
            if field.name not in mapping.keys():
                raise ValueError(f'Unrecognized block: {field.name}')
            return v

        @root_validator(pre=True, allow_reuse=True)
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
    def parse_data_with_class(cls, content, output_class_name, output_data_mapping):
        output_class = ActionOutput.create_model_class(output_class_name, output_data_mapping)
        parsed_data = OutputParser.parse_data_with_mapping(content, output_data_mapping)
        logger.debug(parsed_data)
        instruct_content = output_class(**parsed_data)
        return instruct_content

    @classmethod
    def preprocess_json_str(cls,json_str: str) -> str:
        # 替换字符串内的实际换行为 \n
        return re.sub(r'(?<=")([^"]*?)(?=")', lambda m: m.group(0).replace("\n", ""), json_str)

