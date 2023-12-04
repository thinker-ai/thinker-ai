from typing import Any

from pydantic import BaseModel


class WorkFlowSchema(BaseModel):
    def __init__(self, id: str, name: str, owner_id: str, **data: Any):
        super().__init__(**data)
        self.id = id
        self.name = name
        self.owner_id = owner_id
