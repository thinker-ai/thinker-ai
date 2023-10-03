from abc import ABC, abstractmethod
from typing import Dict

from pydantic import BaseModel

from thinker_ai.actions.action import Criteria
from thinker_ai.agent.worker_factory import WorkerFactory


class Task(BaseModel):
    worker_factory: WorkerFactory
    solution_id: str
    task_id: str
    input_file: str
    max_try: str
    criteria: Criteria

    def __init__(self, id: str, name: str, role_name: str, *args, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.name = name
        self.role_name = role_name

    @abstractmethod
    def do(self):
        worker = self.worker_factory.get_worker(self.role_name)
        worker.work(self)


class CompositeTask(Task, ABC):
    task_infos: Dict[str, Task] = {}

    def __init__(self, id: str, name: str):
        super().__init__(id, name)

    def add_task_info(self, task_info: Task):
        self.task_infos[task_info.name] = task_info
