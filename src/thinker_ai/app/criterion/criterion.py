from pydantic import BaseModel


class Criterion(BaseModel):
    name:str
    description:str