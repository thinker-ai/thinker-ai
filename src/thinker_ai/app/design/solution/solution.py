from typing import Set

from pydantic import BaseModel

from thinker_ai.app.criterion.criterion import Criterion
from thinker_ai.app.design.solution.solution_node import SolutionTreeNode
from thinker_ai.app.mindset.mindset import Mindset
from thinker_ai.common.resource import Resource


class Solution(Resource, BaseModel):
    user_id: str
    name: str = ""
    description: str = ""
    solution_tree: SolutionTreeNode = None
    resources: Set[Resource] = []
    mindsets: Set[Mindset] = []
    criterion: Set[Criterion] = []

    class Config:
        arbitrary_types_allowed = True
