import uuid
from typing import Set, Union, Any

from pydantic import BaseModel

from thinker_ai.app.criterion.criterion import Criterion
from thinker_ai.app.design.solution.solution_node_repository import state_machine_definition_repository
from thinker_ai.app.design.solution.solution_tree_node_facade import SolutionTreeNodefacade
from thinker_ai.app.mindset.mindset import Mindset
from thinker_ai.common.resource import Resource
from thinker_ai.status_machine.state_machine_definition import StateMachineDefinition, StateMachineDefinitionBuilder, \
    StateDefinition, CompositeStateDefinition

facade = SolutionTreeNodefacade()


class Solution(Resource, BaseModel):
    id: str
    user_id: str
    name: str = None
    description: str = None

    resources: Set[Resource] = []
    mindsets: Set[Mindset] = []
    criterion: Set[Criterion] = []

    class Config:
        arbitrary_types_allowed = True

    @property
    async def solution_tree(self) -> Union[StateMachineDefinition, str]:
        if self.name:
            result = state_machine_definition_repository.get_root(self.id)
            if result is None and self.description is not None:
                result = await self.generate_state_machine_definition(state_machine_definition_name=self.name,
                                                                      description=self.description)
            return result

    async def generate_state_machine_definition(self, state_machine_definition_name: str, description: str) -> Union[
        StateMachineDefinition, str]:
        if self.description is not None:
            result = await facade.try_plan(group_id=self.id,
                                           state_machine_definition_name=state_machine_definition_name,
                                           description=description,
                                           max_retry=3)
            if result.is_success:
                return result.state_machine_definition
            else:
                return result.message
        return ""

    async def to_dict(self) -> dict:
        result = dict()
        result['id'] = self.id
        result['user_id'] = self.user_id
        result['name'] = self.name
        result['description'] = self.description
        result['solution_tree'] = self.build_menu_tree(await self.solution_tree)
        result['resources'] = self.resources
        result['mindsets'] = self.mindsets
        result['criterion'] = self.criterion
        return result

    def build_menu_tree(self, solution_tree) -> list:
        menu_tree = list()
        if not solution_tree:
            return menu_tree
        if isinstance(solution_tree, str):
            menu_tree.append(solution_tree)
        else:
            states_def = solution_tree.states_def
            for child in states_def:
                if isinstance(child, StateDefinition):
                    child_node = {
                        "name": child.label,
                        "description": child.description,
                    }
                    if isinstance(child, CompositeStateDefinition):
                        children_tree = self.build_menu_tree(child.state_machine_definition)
                        child_node["children"] = children_tree
                    else:
                        child_node["children"] = []
                    menu_tree.append(child_node)
        return menu_tree
