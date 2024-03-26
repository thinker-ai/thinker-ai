from pydantic import BaseModel
from typing import TypeVar, Type
from fastapi.encoders import jsonable_encoder
import json

# 定义一个类型变量，用于类型提示
T = TypeVar('T', bound='Serializable')

class Serializable(BaseModel):
    def serialize(self) -> str:
        json_compatible_item_data = jsonable_encoder(self)
        return json.dumps(json_compatible_item_data, ensure_ascii=False)

    @classmethod
    def deserialize(cls: Type[T], json_str: str) -> T:
        # 注意这里使用 cls 来代替直接的 Serializable，以确保返回正确的类型
        return cls.parse_raw(json_str)
