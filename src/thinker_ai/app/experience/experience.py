from pydantic import BaseModel


class Experience(BaseModel):
    name: str
    description: str