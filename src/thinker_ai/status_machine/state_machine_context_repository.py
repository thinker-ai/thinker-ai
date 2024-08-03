import json
import os
from json import JSONDecodeError

from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_context import StateMachineContextBuilder, StateMachine, \
    StateMachineRepository


class DefaultStateMachineContextRepository(StateMachineRepository):

    def __init__(self, state_machine_builder: StateMachineContextBuilder,
                 state_machine_def_repo: StateMachineDefinitionRepository,
                 contexts: dict = None):
        if not contexts:
            self.contexts = {}
        else:
            self.contexts = contexts
        self.state_machine_builder = state_machine_builder
        self.state_machine_def_repo = state_machine_def_repo

    @classmethod
    def from_file(cls, base_dir: str, file_name: str, state_machine_builder: StateMachineContextBuilder,
                  state_machine_def_repo: StateMachineDefinitionRepository) -> "DefaultStateMachineContextRepository":
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    contexts = json.load(file)
                except JSONDecodeError:
                    contexts = {}
                return cls(state_machine_builder=state_machine_builder,
                           state_machine_def_repo=state_machine_def_repo,
                           contexts=contexts)

    @classmethod
    def new(cls, state_machine_builder: StateMachineContextBuilder,
            state_machine_def_repo: StateMachineDefinitionRepository
            ) -> "DefaultStateMachineContextRepository":
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo)

    @classmethod
    def form_json(cls, state_machine_builder: StateMachineContextBuilder,
                  state_machine_def_repo: StateMachineDefinitionRepository,
                  json_text: str) -> "DefaultStateMachineContextRepository":
        if json_text:
            contexts: dict = json.loads(json_text)
        else:
            contexts = {}
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo,
                   contexts=contexts)

    def group_to_json(self,context_group_id: str) -> str:
        group_data = self.contexts.get(context_group_id)
        if group_data:
            return json.dumps(group_data, indent=2, ensure_ascii=False)
        else:
            return ""

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(self.contexts, file, indent=2)

    def set(self, context_group_id: str, state_machine_context: StateMachine):
        group_data = self.contexts.get(context_group_id)
        if not group_data:
            group_data={}
            self.contexts[context_group_id]=group_data
        group_data[state_machine_context.context_id] = self.state_machine_builder.state_machine_to_dict(state_machine_context)

    def get(self, context_group_id: str, context_id: str) -> StateMachine:
        group_data = self.contexts.get(context_group_id)
        if group_data:
            data = group_data.get(context_id)
            if data:
                return self.state_machine_builder.state_machine_from_dict(context_group_id,context_id,
                                                                          data,
                                                                          self.state_machine_def_repo,
                                                                          self)
