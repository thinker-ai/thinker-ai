from pydantic import BaseModel


class Mindset(BaseModel):
    name: str
    description: str