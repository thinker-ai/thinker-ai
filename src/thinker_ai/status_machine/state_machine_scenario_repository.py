import json
import os
from json import JSONDecodeError

from thinker_ai.status_machine.state_machine_definition import StateMachineDefinitionRepository
from thinker_ai.status_machine.state_machine_scenario import StateMachineScenarioBuilder, StateMachineScenario, \
    StateMachineScenarioRepository


class DefaultStateMachineScenarioRepository(StateMachineScenarioRepository):

    def __init__(self, state_machine_builder: StateMachineScenarioBuilder,
                 state_machine_def_repo: StateMachineDefinitionRepository,
                 scenarios: dict = None,
                 base_dir: str = None,
                 file_name: str = None):
        self.base_dir = base_dir
        self.file_name = file_name
        if not scenarios:
            self.scenarios = {}
        else:
            self.scenarios = scenarios
        self.state_machine_builder = state_machine_builder
        self.state_machine_def_repo = state_machine_def_repo

    @classmethod
    def from_file(cls, base_dir: str, file_name: str, state_machine_builder: StateMachineScenarioBuilder,
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
    def new(cls, state_machine_builder: StateMachineScenarioBuilder,
            state_machine_def_repo: StateMachineDefinitionRepository
            ) -> "DefaultStateMachineScenarioRepository":
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo)

    @classmethod
    def form_json(cls, state_machine_builder: StateMachineScenarioBuilder,
                  state_machine_def_repo: StateMachineDefinitionRepository,
                  json_text: str) -> "DefaultStateMachineScenarioRepository":
        if json_text:
            scenarios: dict = json.loads(json_text)
        else:
            scenarios = {}
        return cls(state_machine_builder=state_machine_builder,
                   state_machine_def_repo=state_machine_def_repo,
                   scenarios=scenarios)

    def group_to_json(self, scenario_group_id: str) -> str:
        group_data = self.scenarios.get(scenario_group_id)
        if group_data:
            return json.dumps(group_data, indent=2, ensure_ascii=False)
        else:
            return ""

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(self.scenarios, file, indent=2)

    def save(self):
        if self.base_dir and self.file_name:
            self.to_file(self.base_dir, self.file_name)
        else:
            raise Exception('No base dir or file_name specified')

    def set_scenario(self, scenario_group_id: str, state_machine_scenario: StateMachineScenario):
        state_machine_scenario_dict = self.state_machine_builder.state_machine_scenario_to_dict(state_machine_scenario)
        self.set_dict(scenario_group_id, state_machine_scenario_dict)

    def set_dict(self, scenario_group_id: str, state_machine_scenario_dict: dict):
        group_data = self.scenarios.get(scenario_group_id)
        if not group_data:
            group_data = {}
            self.scenarios[scenario_group_id] = group_data
        group_data.update(state_machine_scenario_dict)

    def get(self, scenario_group_id: str, scenario_id: str) -> StateMachineScenario:
        group_data = self.scenarios.get(scenario_group_id)
        if group_data:
            data = group_data.get(scenario_id)
            return self.state_machine_builder.state_machine_scenario_from_dict(scenario_group_id, scenario_id,
                                                                               data,
                                                                               self.state_machine_def_repo,
                                                                               self)

