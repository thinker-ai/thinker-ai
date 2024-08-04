import json
from typing import Type, Optional, Tuple, List

from thinker_ai.agent.actions.state_machine_task.task_actions import AskPlanResult, PlanAction, AskReview, ReviewConst
from thinker_ai.agent.actions.state_machine_task.task_builder import TaskBuilder
from thinker_ai.agent.actions.state_machine_task.task_repository import state_machine_definition_repository, \
    state_machine_scenario_repository
from thinker_ai.agent.memory.memory import Memory
from thinker_ai.agent.provider.schema import Message
from thinker_ai.common.logs import logger
from thinker_ai.status_machine.base import Command
from thinker_ai.status_machine.state_machine_definition import StateDefinition
from thinker_ai.status_machine.state_machine_scenario import CompositeStateScenario, StateMachineContextBuilder


class CompositeTask(CompositeStateScenario):
    def __init__(self,
                 scenario_id: str,
                 scenario_group_id: str,
                 state_def: StateDefinition,
                 plan_update_max_retry: int = 3,
                 human_confirm: bool = False,
                 ):
        super().__init__(scenario_id, scenario_group_id, state_def, TaskBuilder,
                         state_machine_scenario_repository, state_machine_definition_repository)
        self.plan_update_max_retry = plan_update_max_retry
        self.plan_action = PlanAction()
        self.exec_logger: Memory = Memory()
        self.human_confirm = human_confirm

    async def plan_and_act(self, plan_update_max_retry: Optional[int] = None,
                           task_execute_max_retry: Optional[int] = None):
        if await self.try_plan(plan_update_max_retry):
            execute_plan_success = await self.try_execute(task_execute_max_retry)
            if execute_plan_success:
                self.task_result = await AskPlanResult().run(self)
            else:
                logger.info("计划执行失败")

    async def try_plan(self, max_retry) -> bool:
        max_retry = max_retry if max_retry else self.plan_update_max_retry
        plan_update_count = 0
        while plan_update_count < max_retry:
            plan_update_count += 1
            if await self.write_plan():
                return True
        logger.info("更新计划次数超限，任务失败")
        return False

    async def write_plan(self) -> bool:
        try:
            rsp_plan = await PlanAction().run(
                goal=self.state_def.group_def_name,
                task_name=self.state_def.name,
                instruction=self.state_def.description,
                exec_logger=self.exec_logger
            )
            success, error = self.pre_check_plan_from_rsp(rsp_plan, self.state_def.group_def_name, self.state_def.name)
            self.exec_logger.add(Message(content=rsp_plan, role="assistant", cause_by=PlanAction))
            if not success:
                error_msg = f"The generated plan is not valid with error: {error}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.error(error_msg)
                self.exec_logger.add(Message(content=error_msg, role="assistant", cause_by=PlanAction))
                return False
            _, pass_review = await self._ask_review(human_confirm=False)
            if not pass_review:
                return False
            else:
                self.save_plan(rsp=rsp_plan)
                self.exec_logger.clear()
                return True
        except Exception as e:
            error_msg = f"计划生成失败:{str(e)}"
            logger.error(error_msg)
            self.exec_logger.add(Message(content=error_msg, role="assistant", cause_by=PlanAction))
            return False

    @classmethod
    def pre_check_plan_from_rsp(cls, rsp: str, goal, task_name) -> Tuple[bool, str]:
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
                    return True, "状态机验证成功。"
                else:
                    return False, str(fail_paths)
        except Exception as e:
            return False, str(e)

    async def _ask_review(self, human_confirm: bool = False, review_context_len: int = 5):
        """
        Ask to review the task result, reviewer needs to provide confirmation or request change.
        If human confirms the task result, then we deem the task completed, regardless of whether the code run succeeds;
        if auto mode, then the code run has to succeed for the task to be considered completed.
        """
        context = self.exec_logger.get()
        human_confirm = human_confirm or self.human_confirm
        if human_confirm:
            review, confirmed = await AskReview().run(
                exec_logs=context, task_desc=self.state_def.task_type.desc, trigger=ReviewConst.TASK_REVIEW_TRIGGER
            )
            if not confirmed:
                self.exec_logger.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        else:
            ## TODO:需要自动审查
            return "", True

    def save_plan(self, rsp: str):
        groups_def_dict_data = json.loads(rsp)
        group_id = next(groups_def_dict_data.keys())
        state_machine_def_dict = groups_def_dict_data.get(group_id)
        self.state_machine_definition_repository.set_dict(group_id, state_machine_def_dict)
        self.state_machine_definition_repository.save()

    async def try_execute(self, task_execute_max_retry):
        pass
