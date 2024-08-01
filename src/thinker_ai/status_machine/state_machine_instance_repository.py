import json
import os
from json import JSONDecodeError

from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_instance import StateMachineInstanceBuilder, StateMachine, \
    StateMachineRepository


class DefaultStateMachineContextRepository(StateMachineRepository):

    def __init__(self, state_machine_builder: StateMachineInstanceBuilder,
                 state_machine_def_repo: StateMachineDefinitionRepository,
                 instances: dict = None):
        if not instances:
            self.instances = {}
        else:
            self.instances = instances
        self.state_machine_builder = state_machine_builder
        self.state_machine_def_repo = state_machine_def_repo

    @classmethod
    def from_file(cls, base_dir: str, file_name: str, state_machine_builder: StateMachineInstanceBuilder,
                  state_machine_def_repo: StateMachineDefinitionRepository) -> "DefaultStateMachineContextRepository":
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    instances = json.load(file)
                except JSONDecodeError:
                    instances = {}
                return cls(state_machine_builder=state_machine_builder,
                           state_machine_def_repo=state_machine_def_repo,
                           instances=instances)

    @classmethod
    def new(cls, state_machine_builder: StateMachineInstanceBuilder,
            state_machine_def_repo: StateMachineDefinitionRepository
            ) -> "DefaultStateMachineContextRepository":
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo)

    @classmethod
    def form_json(cls, state_machine_builder: StateMachineInstanceBuilder,
                  state_machine_def_repo: StateMachineDefinitionRepository,
                  json_text: str) -> "DefaultStateMachineContextRepository":
        if json_text:
            instances: dict = json.loads(json_text)
        else:
            instances = {}
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo,
                   instances=instances)

    def to_json(self) -> str:
        return json.dumps(self.instances, indent=2, ensure_ascii=False)

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(self.instances, file, indent=2)

    def set(self, group_id: str, state_machine_instance: StateMachine):
        group_data = self.instances.get(group_id)
        if not group_data:
            group_data={}
            self.instances[group_id]=group_data
        group_data[state_machine_instance.id] = self.state_machine_builder.state_machine_to_dict(state_machine_instance)

    def get(self, group_id: str, instance_id: str) -> StateMachine:
        group_data = self.instances.get(group_id)
        if group_data:
            data = group_data.get(instance_id)
            if data:
                return self.state_machine_builder.state_machine_from_dict(group_id,instance_id,
                                                                          data,
                                                                          self.state_machine_def_repo,
                                                                          self)
