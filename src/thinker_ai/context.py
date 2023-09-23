from pydantic import BaseModel

from thinker_ai.actor import Actor


class Context:
    def __init__(self,organization_id:str,actor:Actor,solution_name:str):
        self.organization_id=organization_id
        self.actor=actor
        self.solution_name=solution_name


