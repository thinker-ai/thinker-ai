from thinker_ai.configs.config import config
from thinker_ai.status_machine.state_machine_scenario import StateMachineScenarioBuilder
from thinker_ai.status_machine.state_machine_scenario_repository import DefaultStateMachineScenarioRepository
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository

state_machine_definition_repository = DefaultBasedStateMachineDefinitionRepository.from_file(
    str(config.workspace.path / "data"),
    config.state_machine.definition)
state_machine_scenario_repository = DefaultStateMachineScenarioRepository.from_file(str(config.workspace.path / "data"),
                                                                                    config.state_machine.scenario,
                                                                                    StateMachineScenarioBuilder(),
                                                                                    state_machine_definition_repository)
