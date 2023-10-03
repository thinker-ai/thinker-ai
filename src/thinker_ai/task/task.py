from abc import ABC, abstractmethod
from typing import Dict

from pydantic import BaseModel

from thinker_ai.role.role_factory import RoleFactory


class Task(BaseModel):
    role_factory: RoleFactory
    solution_id: str
    task_id: str
    input_file: str

    def __init__(self, id: str, name: str, role_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id
        self.name = name
        self.role_name = role_name

    @abstractmethod
    def do(self):
        role = self.role_factory.get_role(self.role_name)
        role.do_task(self)


class CompositeTask(Task, ABC):
    task_infos: Dict[str, Task] = {}

    def __init__(self, id: str, name: str):
        super().__init__(id, name)

    def add_task_info(self, task_info: Task):
        self.task_infos[task_info.name] = task_info
