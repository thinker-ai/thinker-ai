import json
from typing import List, Tuple

from thinker_ai.agent.memory.memory import Memory
from thinker_ai.agent.provider.schema import Message
from thinker_ai.app.design.solution.ai_actions import StateMachineDefinitionAction
from thinker_ai.app.design.solution.solution_node import SolutionResult, PlanResult
from thinker_ai.app.design.solution.solution_node_repository import state_machine_definition_repository, \
    state_machine_scenario_repository
from thinker_ai.common.logs import logger
from thinker_ai.status_machine.base import Command
from thinker_ai.status_machine.state_machine_scenario import StateMachineScenarioBuilder


class SolutionTreeNodefacade:

    def __init__(self,
                 human_confirm=True,
                 plan_update_max_retry: int = 3):
        self.plan_update_max_retry = plan_update_max_retry
        self.plan_action = StateMachineDefinitionAction()
        self.exec_logger: Memory = Memory()
        self.human_confirm = human_confirm

    async def try_plan(self,
                       group_id: str,
                       state_machine_definition_name: str,
                       description: str,
                       max_retry: int) -> PlanResult:

        max_retry = max_retry if max_retry else self.plan_update_max_retry
        plan_update_count = 0
        while plan_update_count < max_retry:
            plan_update_count += 1
            state_machine_definition = await self._create_state_machine_definition(group_id=group_id,
                                                                                   state_machine_definition_name=state_machine_definition_name,
                                                                                   description=description
                                                                                   )
            if state_machine_definition:
                return state_machine_definition
        error_msg = "生成次数超限，任务失败"
        logger.info(error_msg)
        return PlanResult(is_success=False, message=error_msg)

    async def _create_state_machine_definition(self,
                                               group_id: str,
                                               state_machine_definition_name: str,
                                               description: str
                                               ) -> PlanResult:
        try:
            rsp_state_machine_definition = await StateMachineDefinitionAction().run(
                group_id=group_id,
                state_machine_definition_name=state_machine_definition_name,
                description=description,
                exec_logger=self.exec_logger,
            )
            plan_result = self._pre_check_plan_from_rsp(rsp_state_machine_definition, group_id, state_machine_definition_name)
            self.exec_logger.add(Message(content=rsp_state_machine_definition, role="assistant", cause_by=StateMachineDefinitionAction))
            if not plan_result.is_success:
                error_msg = f"The generated plan is not valid with error: {plan_result.message}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.error(error_msg)
                self.exec_logger.add(
                    Message(content=error_msg, role="assistant", cause_by=StateMachineDefinitionAction))
                return PlanResult(is_success=False, message=error_msg)
            else:
                self._save_plan(rsp=rsp_state_machine_definition)
                self.exec_logger.clear()
                return plan_result
        except Exception as e:
            error_msg = f"计划生成失败:{str(e)}"
            logger.error(error_msg)
            self.exec_logger.add(Message(content=error_msg, role="assistant", cause_by=StateMachineDefinitionAction))
            return PlanResult(is_success=False, message=error_msg)

    @staticmethod
    def _pre_check_plan_from_rsp(rsp: str, group_id, state_machine_definition_name) -> PlanResult:
        try:
            state_machine = (StateMachineScenarioBuilder
                             .new_from_group_def_json(state_machine_def_group_id=group_id,
                                                      state_machine_def_name=state_machine_definition_name,
                                                      def_json=rsp,
                                                      state_machine_definition_repository=state_machine_definition_repository,
                                                      state_machine_scenario_repository=state_machine_scenario_repository)
                             )

            if state_machine:
                results: List[Tuple[List[Command], bool]] = state_machine.self_validate()
                success = True
                fail_paths = []
                for command_list, result in results:
                    if not result:
                        fail_path = []
                        for command in command_list:
                            fail_path.append((command.name, command.target))
                            fail_paths.append(fail_path)
                        success = result
                if success:
                    return PlanResult(is_success=True, state_machine_definition=state_machine.get_state_machine_def())
                else:
                    return PlanResult(is_success=False, message=str(fail_paths))
        except Exception as e:
            return PlanResult(is_success=False, message=str(e))

    @staticmethod
    def _save_plan(rsp: str):
        groups_def_dict_data = json.loads(rsp)
        group_id = next(iter(groups_def_dict_data.keys()))
        state_machine_def_dict = groups_def_dict_data.get(group_id)
        state_machine_definition_repository.set_dict(group_id, state_machine_def_dict)
        state_machine_definition_repository.save()

    async def try_execute(self, task_execute_max_retry) -> SolutionResult:
        pass
