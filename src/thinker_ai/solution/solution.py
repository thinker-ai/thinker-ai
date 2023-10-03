from abc import ABC
from typing import Dict

from pydantic import BaseModel

from thinker_ai.task.task import Task, Task


class Solution(BaseModel):
    customer_id: str
    id: str
    name: str
    solution_home: str
    tasks: Dict[str, Task] = {}

    def __init__(self, id: str, customer_id: str, name: str, solution_home: str, tasks: Dict[str, Task], *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id
        self.name = name
        self.customer_id = customer_id
        self.solution_home = solution_home
        self.tasks = tasks

    def add_task_info(self, task: Task):
        self.tasks[task.name] = task
