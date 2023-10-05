from thinker_ai.solution.task import Task


class TaskSchema:

    def __init__(self,task:Task):
        self.task=task


class TaskFlow:
    def __init__(self,name:str):
        self.name=name