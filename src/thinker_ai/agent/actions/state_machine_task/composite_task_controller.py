import json
from typing import List, Tuple

from thinker_ai.agent.actions.state_machine_task.ai_actions import PlanAction
from thinker_ai.agent.actions.state_machine_task.composite_task import PlanResult
from thinker_ai.agent.actions.state_machine_task.composite_task_ui import CompositeTaskUI
from thinker_ai.agent.actions.state_machine_task.task import TaskResult
from thinker_ai.agent.actions.state_machine_task.task_repository import state_machine_definition_repository, \
    state_machine_scenario_repository
from thinker_ai.agent.memory.memory import Memory
from thinker_ai.agent.provider.schema import Message
from thinker_ai.common.logs import logger
from thinker_ai.status_machine.base import Command
from thinker_ai.status_machine.state_machine_scenario import StateMachineContextBuilder


class CompositeTaskDefinitionController:

    def __init__(self,
                 human_confirm=True,
                 plan_update_max_retry: int = 3):
        self.plan_update_max_retry = plan_update_max_retry
        self.plan_action = PlanAction()
        self.exec_logger: Memory = Memory()
        self.human_confirm = human_confirm
    async def try_plan(self,
                       goal: str,
                       task_name: str,
                       instruction: str,
                       max_retry) -> CompositeTaskUI:

        max_retry = max_retry if max_retry else self.plan_update_max_retry
        plan_update_count = 0
        while plan_update_count < max_retry:
            plan_update_count += 1
            plan_result=await self._write_plan(goal, task_name, instruction)
            if plan_result:
                return CompositeTaskUI(plan_result)
        error_msg = "生成计划次数超限，任务失败"
        logger.info(error_msg)
        return CompositeTaskUI(PlanResult(is_success=False, message=error_msg))

    async def _write_plan(self,
                          goal: str,
                          task_name: str,
                          instruction: str,
                          ) -> PlanResult:
        try:
            rsp_plan = await PlanAction().run(
                goal=goal,
                task_name=task_name,
                instruction=instruction,
                exec_logger=self.exec_logger
            )
            plan_result = self._pre_check_plan_from_rsp(rsp_plan, goal, task_name)
            self.exec_logger.add(Message(content=rsp_plan, role="assistant", cause_by=PlanAction))
            if not plan_result.is_success:
                error_msg = f"The generated plan is not valid with error: {plan_result.message}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.error(error_msg)
                self.exec_logger.add(Message(content=error_msg, role="assistant", cause_by=PlanAction))
                return PlanResult(is_success=False, message=error_msg)
            else:
                self._save_plan(rsp=rsp_plan)
                self.exec_logger.clear()
                return plan_result
        except Exception as e:
            error_msg = f"计划生成失败:{str(e)}"
            logger.error(error_msg)
            self.exec_logger.add(Message(content=error_msg, role="assistant", cause_by=PlanAction))
            return PlanResult(is_success=False, message=error_msg)

    @staticmethod
    def _pre_check_plan_from_rsp(rsp: str, goal, task_name) -> PlanResult:
        try:
            state_machine = (StateMachineContextBuilder
                             .new_from_group_def_json(state_machine_def_group_name=goal,
                                                      state_machine_def_name=task_name,
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
        group_id = next(groups_def_dict_data.keys())
        state_machine_def_dict = groups_def_dict_data.get(group_id)
        state_machine_definition_repository.set_dict(group_id, state_machine_def_dict)
        state_machine_definition_repository.save()

    async def try_execute(self, task_execute_max_retry) -> TaskResult:
        pass
