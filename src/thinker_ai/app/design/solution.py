from typing import Set

from pydantic import BaseModel

from thinker_ai.agent.tools.tool_data_type import Tool
from thinker_ai.app.criterion.criterion import Criterion
from thinker_ai.app.design.solution_nodes import SolutionTreeNode
from thinker_ai.app.experience.experience import Experience
from thinker_ai.app.mindset.mindset import Mindset
from thinker_ai.common.resource import Resource


class Solution(Resource,BaseModel):
    title: str
    description: str
    solution_tree: SolutionTreeNode
    resources: Set[Resource]
    experiences: Set[Experience]
    mindsets: Set[Mindset]
    criterion: Set[Criterion]
