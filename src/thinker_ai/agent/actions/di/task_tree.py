from __future__ import annotations

from overrides import overrides

import json
from typing import Tuple, Optional, List, Any

from thinker_ai.agent.actions import Action
from thinker_ai.status_machine.base import Command
from thinker_ai.status_machine.state_machine_instance import StateMachineInstanceBuilder, DefaultStateContextBuilder
from thinker_ai.status_machine.task_desc import TaskType, TaskDesc, PlanStatus, TaskTypeDef
from thinker_ai.common.common import replace_curly_braces
from thinker_ai.configs.config import config
from thinker_ai.status_machine.state_machine_instance_repository import DefaultStateMachineContextRepository
from thinker_ai.status_machine.status_machine_definition_repository import DefaultBasedStateMachineDefinitionRepository
from thinker_ai.utils.code_parser import CodeParser
from thinker_ai.agent.actions.di.task import Task, AskReview, ReviewConst, exec_logger, tasks_storage, code_executor, \
    TaskResult
from thinker_ai.agent.provider.schema import Message
from thinker_ai.common.logs import logger
from thinker_ai.common.val_class import remove_comments

DATA_INFO: str = """
# Latest Data Info
Latest data info after previous tasks:
{info}
"""
definition_repo = DefaultBasedStateMachineDefinitionRepository.from_file(str(config.workspace.path / "data"),
                                                                         config.state_machine.definition)
instance_repo = DefaultStateMachineContextRepository.from_file(str(config.workspace.path / "data"),
                                                               config.state_machine.instance,
                                                               StateMachineInstanceBuilder(),
                                                               definition_repo)


async def confirm_review(review: str) -> bool:
    confirmed_and_more = (
            ReviewConst.CONTINUE_WORDS[0] in review.lower() and review.lower() not in ReviewConst.CONTINUE_WORDS[0]
    )  # "confirm, ... (more content, such as changing downstream tasks)"
    if confirmed_and_more or "redo" in review:
        exec_logger.add(Message(content=review, role="user", cause_by=AskReview))
        return False
    return True


class TaskTree(Task):
    tasks: List[Task] = []
    task_map: dict[str, Task] = {}
    current_task_id: Optional[str] = None
    plan_update_max_retry: int = 3
    type: TaskTypeDef = TaskType.STATE_MACHINE_PLAN.value

    def __init__(self,
                 tasks: Optional[List[Task]] = None, **kwargs):
        super().__init__(**kwargs)
        self.add_tasks(tasks)

    async def plan_and_act(self, plan_update_max_retry: Optional[int] = None,
                           task_execute_max_retry: Optional[int] = None):
        if not tasks_storage.load(self.id):
            tasks_storage.save(self)
        if await self.try_plan(plan_update_max_retry):
            execute_plan_success = await self.try_act(task_execute_max_retry)
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

    async def try_act(self, max_retry: Optional[int] = None) -> bool:
        max_retry = max_retry if max_retry else self.task_execute_max_retry
        current_task_retry_counter = 0
        while self.current_task and current_task_retry_counter < max_retry:
            current_task_retry_counter += 1
            await self._check_data()
            await self.current_task.write_and_exec_code(first_trial=current_task_retry_counter == 1)
            current_result = self.current_task.task_result
            pass_review = False
            if current_result and current_result.is_success:
                pass_review = await self.review_current_result()
            if pass_review:
                self._update_current_task()
                current_task_retry_counter = 0

        if len(self.tasks) == len(self.get_finished_tasks()):
            return True
        else:
            return False

    @property
    def task_desc(self) -> TaskDesc:
        task_desc: TaskDesc = super().task_desc
        task_desc.map["plan"] = self.plan_written
        return task_desc

    @property
    def plan_written(self) -> list[dict]:
        plan_tasks = []
        for task in self.tasks:
            plan_tasks.append(task.task_desc.map)
        return plan_tasks

    @property
    def code_written(self) -> dict[str, str]:
        finished_tasks = self.get_finished_tasks()
        return {task.id: remove_comments(task.task_result.code) for task in finished_tasks}

    @property
    def exec_results(self) -> dict[str, str]:
        finished_tasks = self.get_finished_tasks()
        return {task.id: task.task_result.result for task in finished_tasks}

    @property
    def plan_status(self) -> PlanStatus:
        return PlanStatus(code_written=self.code_written, exec_results=self.exec_results)

    @property
    def current_task(self) -> Task:
        """Find current task to execute

        Returns:
            Task: the current task to be executed
        """
        return self.task_map.get(self.current_task_id, None)

    async def write_plan(self) -> bool:
        try:
            rsp_plan = await WritePlan().run(
                goal=self.goal,
                task_name=self.name,
                instruction=self.instruction
            )
            success, error = precheck_update_plan_from_rsp(rsp_plan, self)
            exec_logger.add(Message(content=rsp_plan, role="assistant", cause_by=WritePlan))
            if not success:
                error_msg = f"The generated plan is not valid with error: {error}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.error(error_msg)
                exec_logger.add(Message(content=error_msg, role="assistant", cause_by=WritePlan))
                return False
            _, pass_review = await self._ask_review(human_confirm=False)
            if not pass_review:
                return False
            else:
                update_plan_from_rsp(rsp=rsp_plan, task_tree=self)
                exec_logger.clear()
                return True
        except Exception as e:
            error_msg = f"计划生成失败:{str(e)}"
            logger.error(error_msg)
            exec_logger.add(Message(content=error_msg, role="assistant", cause_by=WritePlan))
            return False

    @overrides
    async def _ask_review(self, human_confirm: bool = False, review_context_len: int = 5):
        """
        Ask to review the task result, reviewer needs to provide confirmation or request change.
        If human confirms the task result, then we deem the task completed, regardless of whether the code run succeeds;
        if auto mode, then the code run has to succeed for the task to be considered completed.
        """
        context = exec_logger.get()
        human_confirm = human_confirm or self.human_confirm
        if human_confirm:
            review, confirmed = await AskReview().run(
                exec_logs=context, task_desc=self.task_desc, trigger=ReviewConst.TASK_REVIEW_TRIGGER
            )
            if not confirmed:
                exec_logger.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        else:
            ## TODO:需要自动审查
            return "", True

    async def review_current_result(self) -> bool:
        # ask for acceptance, users can other refuse and change tasks in the plan
        review, task_result_confirmed = await self._ask_review()
        return task_result_confirmed and await confirm_review(review)

    @overrides
    def to_dict(self):
        task_dict = super().to_dict()
        task_dict['tasks'] = {key: task.to_dict() for key, task in self.task_map.items()}
        return task_dict

    def _topological_sort(self, tasks: list[Task]) -> list[Task]:
        task_map = {task.id: task for task in tasks}
        dependencies = {task.id: set(task.dependent_task_ids) for task in tasks}
        sorted_tasks = []
        visited = set()

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dependent_id in dependencies.get(task_id, []):
                visit(dependent_id)
            sorted_tasks.append(task_map[task_id])

        for task in tasks:
            visit(task.id)

        return sorted_tasks

    def _inherit_form_parent(self, task: Task):
        if not task:
            return
        task.set_tools(self.tools, self.tool_recommender)
        task.human_confirm = self.human_confirm
        task.use_reflection = self.use_reflection
        task.task_execute_max_retry = self.task_execute_max_retry

    def add_tasks(self, tasks: list[Task]):
        """
        Integrates new tasks into the existing plan, ensuring dependency order is maintained.

        This method performs two primary functions based on the current state of the task list:
        1. If there are no existing tasks, it topologically sorts the provided tasks to ensure
        correct execution order based on dependencies, and sets these as the current tasks.
        2. If there are existing tasks, it merges the new tasks with the existing ones. It maintains
        any common prefix of tasks (based on task_id and instruction) and appends the remainder
        of the new tasks. The current task is updated to the first unfinished task in this merged list.

        Args:
            tasks (list[Task]): A list of tasks (may be unordered) to add to the plan.

        Returns:
            None: The method updates the internal state of the plan but does not return anything.
        """
        if not tasks:
            return
        for task in tasks:
            self._inherit_form_parent(task)
        # Topologically sort the new tasks to ensure correct dependency order
        new_tasks = self._topological_sort(tasks)

        if not self.tasks:
            # If there are no existing tasks, set the new tasks as the current tasks
            self.tasks = new_tasks

        else:
            # Find the length of the common prefix between existing and new tasks
            prefix_length = 0
            for old_task, new_task in zip(self.tasks, new_tasks):
                if old_task.id != new_task.id or old_task.task_desc != new_task.task_desc:
                    break
                prefix_length += 1

            # Combine the common prefix with the remainder of the new tasks
            final_tasks = self.tasks[:prefix_length] + new_tasks[prefix_length:]
            self.tasks = final_tasks

        # Update current_task_id to the first unfinished task in the merged list
        self._update_current_task()

        # Update the task map for quick access to tasks by ID
        self.task_map = {task.id: task for task in self.tasks}

    def reset_task(self, task_id: str):
        """
        Clear code and result of the task based on task_id, and set the task as unfinished.

        Args:
            task_id (str): The ID of the task to be reset.

        Returns:
            None
        """
        if task_id in self.task_map:
            task = self.task_map[task_id]
            task.reset()

    def replace_task(self, new_task: Task):
        """
        Replace an existing task with the new input task based on task_id, and reset all tasks depending on it.

        Args:
            new_task (Task): The new task that will replace an existing one.

        Returns:
            None
        """
        assert new_task.id in self.task_map
        # Replace the task in the task map and the task list
        self._inherit_form_parent(new_task)
        self.task_map[new_task.id] = new_task
        for i, task in enumerate(self.tasks):
            if task.id == new_task.id:
                self.tasks[i] = new_task
                break

        # Reset dependent tasks
        for task in self.tasks:
            if new_task.id in task.dependent_task_ids:
                self.reset_task(task.id)

    def append_task(self, new_task: Task):
        """
        Append a new task to the end of existing task sequences

        Args:
            new_task (Task): The new task to be appended to the existing task sequence

        Returns:
            None
        """
        assert not self.has_task_id(new_task.id), "Task already in current plan, use replace_task instead"

        assert all(
            [self.has_task_id(dep_id) for dep_id in new_task.dependent_task_ids]
        ), "New task has unknown dependencies"

        # Existing tasks do not depend on the new task, it's fine to put it to the end of the sorted task sequence
        self._inherit_form_parent(new_task)
        self.tasks.append(new_task)
        self.task_map[new_task.id] = new_task
        self._update_current_task()

    def has_task_id(self, task_id: str) -> bool:
        return task_id in self.task_map

    def _update_current_task(self):
        old_task_id = self.current_task_id
        for task in self.tasks:
            if not task.is_finished:
                self.current_task_id = task.id
                break
        if self.current_task_id == old_task_id:
            self.current_task_id = None

    def get_finished_tasks(self) -> list[Task]:
        """return all finished tasks in correct linearized order

        Returns:
            list[Task]: list of finished tasks
        """
        return [task for task in self.tasks if task.is_finished]

    async def _check_data(self):
        if (  # 这个方法只对DATA_PREPROCESS、FEATURE_ENGINEERING、MODEL_TRAIN的任务类型有效，且必须有finished_tasks
                not self.get_finished_tasks()
                or self.current_task.type
                not in [
            TaskType.DATA_PREPROCESS.type_name,
            TaskType.FEATURE_ENGINEERING.type_name,
            TaskType.MODEL_TRAIN.type_name,
        ]
        ):
            return
        logger.info("Check updated data")
        code = await CheckData().run(self.code_written)
        if not code.strip():
            return
        result, success = await code_executor.run(code)
        if success:
            print(result)
            data_info = DATA_INFO.format(info=result)
            exec_logger.add(Message(content=data_info, role="user", cause_by=CheckData))


def update_state_flow_plan_from_rsp(rsp: str, task_tree: TaskTree):
    tasks = [Task(**task_config) for task_config in rsp]

    if len(tasks) == 1 or tasks[0].dependent_task_ids:
        if tasks[0].dependent_task_ids and len(tasks) > 1:
            # tasks[0].dependent_task_ids means the generated tasks are not a complete plan
            # for they depend on tasks in the current plan, in this case, we only support updating one task each time
            logger.warning(
                "Current plan will take only the first generated task if the generated tasks are not a complete plan"
            )
        # handle a single task
        if task_tree.has_task_id(tasks[0].id):
            # replace an existing task
            task_tree.replace_task(tasks[0])
        else:
            # append one task
            task_tree.append_task(tasks[0])

    else:
        # add tasks in general
        task_tree.add_tasks(tasks)


def update_plan_from_rsp(rsp: str, task_tree: TaskTree):
    state_machine_def = StateMachineInstanceBuilder.state_machine_from_json(rsp,
                                                                            state_machine_definition_repository=definition_repo,
                                                                            state_machine_context_repository=instance_repo)
    tasks = [Task(**task_config) for task_config in rsp]
    if len(tasks) == 1 or tasks[0].dependent_task_ids:
        if tasks[0].dependent_task_ids and len(tasks) > 1:
            # tasks[0].dependent_task_ids means the generated tasks are not a complete plan
            # for they depend on tasks in the current plan, in this case, we only support updating one task each time
            logger.warning(
                "Current plan will take only the first generated task if the generated tasks are not a complete plan"
            )
        # handle a single task
        if task_tree.has_task_id(tasks[0].id):
            # replace an existing task
            task_tree.replace_task(tasks[0])
        else:
            # append one task
            task_tree.append_task(tasks[0])

    else:
        # add tasks in general
        task_tree.add_tasks(tasks)


def precheck_update_plan_from_rsp(rsp: str, task_tree: TaskTree) -> Tuple[bool, str]:
    try:
        state_machine = StateMachineInstanceBuilder.state_machine_from_json(task_tree.goal, task_tree.name, rsp,
                                                                            state_machine_definition_repository=definition_repo,
                                                                            state_machine_context_repository=instance_repo)
        results: List[Tuple[List[Command], bool]] = state_machine.self_validate()
        success = True
        fail_paths = []
        for command_list, result in results:
            if not result:
                fail_path=[]
                for command in command_list:
                    fail_path.append((command.name,command.target))
                    fail_paths.append(fail_path)
                success = result
        if success:
            return True, "状态机验证成功。"
        else:
            return False, str(fail_paths)
    except Exception as e:
        return False, str(e)


class WritePlan(Action):

    def __init__(self, **data: Any):
        super().__init__(**data)

    async def run(self, goal: str, task_name, instruction: str) -> str:
        if not task_name or not instruction:
            raise Exception("task_name and instruction must be provided")
        create_or_update = "create"
        if not goal:
            goal = task_name
        state_machine_def = definition_repo.get(goal, task_name)
        if state_machine_def:
            create_or_update = "update"

        PROMPT_TEMPLATE: str = """
        #The Name Of State Machine Group:
        {goal}
        #The Name Of State Machine:
        {plan_name}
        # Create Or Update
         you will {create_or_update} this State Machine
        # Instruction:
        {instruction}
        # Context:
        {context}
        # Available Task Types:
        {task_type_desc}
        # Guidance:
        the exist state machine definitions are here,they are related to the current state machine and are the context of the current state machine:
        ```json
         {exist_status_definition}
        ```
        the exist state machine instances are here,they are related to the current state machine and are the context of the current state machine:
        ```json
         {exist_status_instance}
        ```
        {guidance}
        """

        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = PROMPT_TEMPLATE.format(
            plan_name=task_name,
            goal=goal,
            create_or_update=create_or_update,
            instruction=instruction,
            context="\n".join([str(ct) for ct in exec_logger.get()]),
            task_type_desc=task_type_desc,
            exist_status_definition=replace_curly_braces(definition_repo.to_json()),
            exist_status_instance=replace_curly_braces(instance_repo.to_json()),
            guidance=TaskType.STATE_MACHINE_PLAN.value.guidance
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp


class CheckData(Action):
    async def run(self, code_written: dict[str, str]) -> str:
        CHECK_DATA_PROMPT = """
        # Background
        Check latest data info to guide subsequent tasks.

        ## Finished Tasks
        ```python
        {code_written}
        ```end

        # Task
        Check code in finished tasks, print key variables to guide your following actions.
        Specifically, if it is a data analysis or machine learning task, print the the latest column information using the following code, with DataFrame variable from 'Finished Tasks' in place of df:
        ```python
        from thinker_ai.agent.tools.libs.data_preprocess import get_column_info

        column_info = get_column_info(df)
        print("column_info")
        print(column_info)
        ```end
        Otherwise, print out any key variables you see fit. Return an empty string if you think there is no important data to check.

        # Constraints:
        - Your code is to be added to a new cell in jupyter.

        # Instruction
        Output code following the format:
        ```python
        your code
        ```
        """
        code_written = "\n".join(code_written.values())
        prompt = CHECK_DATA_PROMPT.format(code_written=code_written)
        rsp = await self._aask(prompt)
        code = CodeParser.parse_code(block=None, text=rsp)
        return code


class AskPlanResult(Action):
    async def run(
            self, task_tree: TaskTree
    ) -> TaskResult:
        PROMPT = """
        # Task Goal
        {user_requirement}
        
        ## Plan Status
        {plan_status}
        
        # Instruction
        Summery the plan execution status to answer the goal result
        """
        user_requirement = task_tree.instruction
        plan_execution_status = [task.to_dict() for task in task_tree.tasks]
        plan_status = json.dumps(obj=plan_execution_status, indent=4, ensure_ascii=False)
        prompt = PROMPT.format(user_requirement=user_requirement, plan_status=plan_status)
        rsp = await self._aask(prompt)
        return TaskResult(is_success=task_tree.is_finished, result=rsp)
