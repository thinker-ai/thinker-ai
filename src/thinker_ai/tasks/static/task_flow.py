from typing import Any

from pydantic import BaseModel

from thinker_ai.app_context import AppContext


class TaskFlow(BaseModel):
    def __init__(self, id: str, name: str, owner_id: str, context: AppContext, work_schema_schema_id: str, **data: Any):
        super().__init__(**data)
        self.id = id
        self.name = name
        self.owner_id = owner_id
        self.context = context
        self.work_schema_schema_id = work_schema_schema_id
