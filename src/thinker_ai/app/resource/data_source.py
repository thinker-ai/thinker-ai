from pydantic import BaseModel

from thinker_ai.common.resource import Resource


class DataSource(Resource, BaseModel):
    path: str
    type: str
