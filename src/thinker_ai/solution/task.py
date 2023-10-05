from abc import abstractmethod, ABC
from typing import Dict

from pydantic import BaseModel

from thinker_ai.actions.action import Criteria
from thinker_ai.agent.worker import WorkerFactory


class Task(BaseModel):
    worker_factory: WorkerFactory
    solution_id: str
    task_id: str
    input_file: str
    max_try: str
    criteria: Criteria

    def __init__(self, id: str, name: str, worker_name: str = None, *args, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.name = name
        self.worker_name = worker_name

    def do(self):
        worker = self.worker_factory.get_worker(self.worker_name)
        worker.work(self)


class CompositeTask(Task, ABC):
    task_infos: Dict[str, Task] = {}

    def __init__(self, id: str, name: str):
        super().__init__(id, name)

    def add_task(self, task: Task):
        self.task_infos[task.name] = task

    @abstractmethod
    def do(self):
        raise NotImplementedError
