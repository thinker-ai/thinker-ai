import uuid

import pytest

from thinker_ai.agent.actions.di.task_tree import (
    PlanTask,
    Task,
    WritePlan,
    validate_state_machine_def,
)
from thinker_ai.configs.config import config
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository


def test_pre_check_update_plan_from_rsp():
    plan = PlanTask(
        id=str(uuid.uuid4()),
        goal="titanic_survival",
        name="titanic_survival",
        instruction="instruction"
    )
    definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(str(config.workspace.path / "data"),
                                                                             config.state_machine.definition)
    rsp = definition_repo.to_json()
    success, _ = validate_state_machine_def(rsp, plan.goal, plan.name)
    assert success
    invalid_rsp = "wrong"
    success, _ = validate_state_machine_def(invalid_rsp, plan.goal, plan.name)
    assert not success


@pytest.mark.asyncio
async def test_write_plan():
    plan = PlanTask(name ="analysis_dataset", goal="data_analysis", role="user")
    rsp = await WritePlan().run(
        goal=plan.root_name,
        task_name=plan.name,
        instruction=plan.instruction
    )

    assert "task_id" in rsp
    assert "instruction" in rsp
