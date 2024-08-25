from typing import Set

from pydantic import BaseModel


class SolutionNode(BaseModel):
    title: str
    description: str
    code_path: str


class SolutionTreeNode(SolutionNode):
    is_root: bool
    children: Set['SolutionNode']
