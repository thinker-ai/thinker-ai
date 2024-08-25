from pydantic import BaseModel

from thinker_ai.common.resource import Resource


class ToolSchema(BaseModel):
    description: str


class Tool(Resource,BaseModel):
    name: str
    description: str = ""
    path: str
    schemas: dict = {}
    code: str = ""
    tags: list[str] = []
