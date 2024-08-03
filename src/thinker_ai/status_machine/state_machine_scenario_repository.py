import json
import os
from json import JSONDecodeError

from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_scenario import StateMachineContextBuilder, StateMachine, \
    StateMachineRepository


class DefaultStateMachineScenarioRepository(StateMachineRepository):

    def __init__(self, state_machine_builder: StateMachineContextBuilder,
                 state_machine_def_repo: StateMachineDefinitionRepository,
                 scenarios: dict = None):
        if not scenarios:
            self.scenarios = {}
        else:
            self.scenarios = scenarios
        self.state_machine_builder = state_machine_builder
        self.state_machine_def_repo = state_machine_def_repo

    @classmethod
    def from_file(cls, base_dir: str, file_name: str, state_machine_builder: StateMachineContextBuilder,
                  state_machine_def_repo: StateMachineDefinitionRepository) -> "DefaultStateMachineScenarioRepository":
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    scenarios = json.load(file)
                except JSONDecodeError:
                    scenarios = {}
                return cls(state_machine_builder=state_machine_builder,
                           state_machine_def_repo=state_machine_def_repo,
                           scenarios=scenarios)

    @classmethod
    def new(cls, state_machine_builder: StateMachineContextBuilder,
            state_machine_def_repo: StateMachineDefinitionRepository
            ) -> "DefaultStateMachineScenarioRepository":
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo)

    @classmethod
    def form_json(cls, state_machine_builder: StateMachineContextBuilder,
                  state_machine_def_repo: StateMachineDefinitionRepository,
                  json_text: str) -> "DefaultStateMachineScenarioRepository":
        if json_text:
            scenarios: dict = json.loads(json_text)
        else:
            scenarios = {}
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo,
                   scenarios=scenarios)

    def group_to_json(self,scenario_group_id: str) -> str:
        group_data = self.scenarios.get(scenario_group_id)
        if group_data:
            return json.dumps(group_data, indent=2, ensure_ascii=False)
        else:
            return ""

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(self.scenarios, file, indent=2)

    def set(self, scenario_group_id: str, state_machine_scenario: StateMachine):
        group_data = self.scenarios.get(scenario_group_id)
        if not group_data:
            group_data={}
            self.scenarios[scenario_group_id]=group_data
        group_data[state_machine_scenario.scenario_id] = self.state_machine_builder.state_machine_to_dict(state_machine_scenario)

    def get(self, scenario_group_id: str, scenario_id: str) -> StateMachine:
        group_data = self.scenarios.get(scenario_group_id)
        if group_data:
            data = group_data.get(scenario_id)
            if data:
                return self.state_machine_builder.state_machine_from_dict(scenario_group_id,scenario_id,
                                                                          data,
                                                                          self.state_machine_def_repo,
                                                                          self)
