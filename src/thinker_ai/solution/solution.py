from typing import Dict

from pydantic import BaseModel

from thinker_ai.solution.task import Task


class Solution(BaseModel):
    customer_id: str
    id: str
    name: str
    solution_home: str
    tasks: Dict[str, Task] = {}

    def __init__(self, id: str, customer_id: str, name: str, solution_home: str, tasks: Dict[str, Task], *args,
                 **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.name = name
        self.customer_id = customer_id
        self.solution_home = solution_home
        self.tasks = tasks

    def add_task_info(self, task: Task):
        self.tasks[task.name] = task

    def run(self):
        raise NotImplementedError



class CompositeSolution(Solution):
    solutions: Dict[str, Solution] = {}

    def __init__(self, id: str, customer_id: str, name: str, solution_home: str, tasks: Dict[str, Task], *args,
                 **kwargs):
        super().__init__(id, customer_id, name, solution_home, tasks, *args, **kwargs)

    def add_solution(self, solution: Solution):
        self.solutions[solution.name] = solution

    def run(self):
        raise NotImplementedError