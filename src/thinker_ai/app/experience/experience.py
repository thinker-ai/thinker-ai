from pydantic import BaseModel

from thinker_ai.common.resource import Resource


class Experience(Resource,BaseModel):
    name: str
    description: str