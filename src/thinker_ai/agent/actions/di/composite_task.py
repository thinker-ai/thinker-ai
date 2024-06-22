from __future__ import annotations

from overrides import overrides

import json
from typing import Tuple, Optional, List

from thinker_ai.agent.actions import Action
from thinker_ai.utils.code_parser import CodeParser
from thinker_ai.agent.actions.di.task import Task, TaskType, AskReview, ReviewConst, CheckData, exec_logger
from thinker_ai.agent.provider.schema import Message
from thinker_ai.common.logs import logger
from thinker_ai.common.val_class import remove_comments


class CompositeTask(Task):
    STRUCTURAL_CONTEXT: str = """
    ## User Requirement
    {user_requirement}
    ## Context
    {context}
    ## Current Plan
    {tasks}
    ## Current Task
    {current_task}
    """
    PLAN_STATUS: str = """
    ## Finished Tasks
    ### code
    ```python
    {code_written}
    ```

    ### execution result
    {task_results}

    ## Current Task
    {current_task}

    ## Task Guidance
    Write complete code for 'Current Task'. And avoid duplicating code from 'Finished Tasks', such as repeated import of packages, reading data, etc.
    Specifically, {guidance}
    """

    DATA_INFO: str = """
    # Latest Data Info
    Latest data info after previous tasks:
    {info}
    """

    tasks: List[Task] = []
    task_map: dict[str, Task] = {}
    current_task_id: str = ""

    def __init__(self, tasks: Optional[List[Task]] = None, **kwargs):
        super().__init__(**kwargs)
        self.task_map = {task.id: task for task in tasks} if tasks else []
        self.tasks = self._topological_sort(tasks if tasks else [])

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def execute_plan(self, instruction):
        await self.update_plan(instruction=instruction)
        # take on tasks until all finished
        while self.current_task:
            # data info
            await self._check_data()
            # take on current task
            await self.current_task.act()
            # process the result, such as reviewing, confirming, plan updating
            await self.process_current_result()
        rsp = self.get_context_msg()[0]  # return the completed plan as a response
        return rsp

    @property
    def plane_status_msg(self) -> str:
        # prepare components of a plan status
        finished_tasks = self.plan.get_finished_tasks()
        code_written = [remove_comments(task.code) for task in finished_tasks]
        code_written = "\n\n".join(code_written)
        task_results = [task.result for task in finished_tasks]
        task_results = "\n\n".join(task_results)
        task_typ_def = TaskType.get_type(self.current_task.task_type)
        guidance = task_typ_def.guidance if task_typ_def else ""

        # combine components in a prompt
        prompt = self.PLAN_STATUS.format(
            code_written=code_written,
            task_results=task_results,
            current_task=self.current_task.instruction,
            guidance=guidance,
        )

        return prompt

    @overrides
    def get_context_msg(self, task_exclude_field=None) -> list[Message]:
        """only to reduce context length and improve performance"""
        tasks = [task.model_dump(exclude=task_exclude_field) for task in self.tasks]
        tasks = json.dumps(tasks, indent=4, ensure_ascii=False)
        context = self.STRUCTURAL_CONTEXT.format(
            user_requirement=self.instruction, context=self.context, tasks=tasks, current_task=self.model_dump_json()
        )
        context_msg = [Message(content=context, role="user")]
        return context_msg + exec_logger.get()

    @property
    def current_task(self) -> Task:
        """Find current task to execute

        Returns:
            Task: the current task to be executed
        """
        return self.task_map.get(self.current_task_id, None)

    @overrides
    async def ask_review(
            self,
            auto_run: bool = None,
            review_context_len: int = 5,
    ):
        auto_run = auto_run or self.auto_run
        if not auto_run:
            context = self.get_context_msg()
            review, confirmed = await AskReview().run(
                context=context[-review_context_len:], composite_task=self, trigger=ReviewConst.TASK_REVIEW_TRIGGER
            )
            if not confirmed:
                exec_logger.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        confirmed = self.task_result.is_success if self.task_result else True
        return "", confirmed

    async def confirm_current_task(self, review: str):
        self._update_current_task()

        confirmed_and_more = (
                ReviewConst.CONTINUE_WORDS[0] in review.lower() and review.lower() not in ReviewConst.CONTINUE_WORDS[0]
        )  # "confirm, ... (more content, such as changing downstream tasks)"
        if confirmed_and_more:
            exec_logger.add(Message(content=review, role="user", cause_by=AskReview))
            await self.update_plan()

    async def update_plan(self, instruction: str = "", max_tasks: int = 3, max_retries: int = 3):
        self.instruction = instruction

        plan_confirmed = False
        rsp_plan = ""
        while not plan_confirmed:
            context = self.get_context_msg()
            rsp_plan = await WritePlan().run(context, max_tasks=max_tasks)
            exec_logger.add(Message(content=rsp_plan, role="assistant", cause_by=WritePlan))

            # precheck plan before asking reviews
            is_plan_valid, error = precheck_update_plan_from_rsp(rsp_plan, self)
            if not is_plan_valid and max_retries > 0:
                error_msg = f"The generated plan is not valid with error: {error}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.warning(error_msg)
                exec_logger.add(Message(content=error_msg, role="assistant", cause_by=WritePlan))
                max_retries -= 1
                continue

            _, plan_confirmed = await self.ask_review(trigger=ReviewConst.TASK_REVIEW_TRIGGER)

        update_plan_from_rsp(rsp=rsp_plan, composite_task=self)

        exec_logger.clear()

    async def process_current_result(self):
        # ask for acceptance, users can other refuse and change tasks in the plan
        review, task_result_confirmed = await self.ask_review()

        if task_result_confirmed:
            # tick off this task and record progress
            await self.confirm_current_task(review)

        elif "redo" in review:
            # Ask the Role to redo this task with help of review feedback,
            # useful when the code run is successful but the procedure or result is not what we want
            pass  # simply pass, not confirming the result

        else:
            # update plan according to user's feedback and to take on changed tasks
            await self.update_plan()

    def _topological_sort(self, tasks: list[Task]):
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

        # Topologically sort the new tasks to ensure correct dependency order
        new_tasks = self._topological_sort(tasks)

        if not self.tasks:
            # If there are no existing tasks, set the new tasks as the current tasks
            self.tasks = new_tasks

        else:
            # Find the length of the common prefix between existing and new tasks
            prefix_length = 0
            for old_task, new_task in zip(self.tasks, new_tasks):
                if old_task.id != new_task.id or old_task.instruction != new_task.instruction:
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
        self.tasks.append(new_task)
        self.task_map[new_task.id] = new_task
        self._update_current_task()

    def has_task_id(self, task_id: str) -> bool:
        return task_id in self.task_map

    def _update_current_task(self):
        for task in self.tasks:
            if not task.is_finished:
                self.current_task_id = task.id
                break

    def get_finished_tasks(self) -> list[Task]:
        """return all finished tasks in correct linearized order

        Returns:
            list[Task]: list of finished tasks
        """
        return [task for task in self.tasks if task.is_finished]

    async def _check_data(self):
        if (  # 这个方法只对DATA_PREPROCESS、FEATURE_ENGINEERING、MODEL_TRAIN的任务类型有效，且必须有finished_tasks
                not self.get_finished_tasks()
                or self.current_task.task_type
                not in [
            TaskType.DATA_PREPROCESS.type_name,
            TaskType.FEATURE_ENGINEERING.type_name,
            TaskType.MODEL_TRAIN.type_name,
        ]
        ):
            return
        logger.info("Check updated data")
        code = await CheckData().run(self.get_finished_tasks())
        if not code.strip():
            return
        result, success = await self.execute_code.run(code)
        if success:
            print(result)
            data_info = self.DATA_INFO.format(info=result)
            exec_logger.add(Message(content=data_info, role="user", cause_by=CheckData))


class WritePlan(Action):
    PROMPT_TEMPLATE: str = """
    # Context:
    {context}
    # Available Task Types:
    {task_type_desc}
    # Task:
    Based on the context, write a plan or modify an existing plan of what you should do to achieve the goal. A plan consists of one to {max_tasks} tasks.
    If you are modifying an existing plan, carefully follow the instruction, don't make unnecessary changes. Give the whole plan unless instructed to modify only one task of the plan.
    If you encounter errors on the current task, revise and output the current single task only.
    Output a list of jsons following the format:
    ```json
    [
        {{
            "task_id": str = "unique identifier for a task in plan, can be an ordinal",
            "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
            "instruction": "what you should do in this task, one short phrase or sentence",
            "task_type": "type of this task, should be one of Available Task Types",
        }},
        ...
    ]
    ```
    """

    async def run(self, context: list[Message], max_tasks: int = 5) -> str:
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = self.PROMPT_TEMPLATE.format(
            context="\n".join([str(ct) for ct in context]), max_tasks=max_tasks, task_type_desc=task_type_desc
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp


def update_plan_from_rsp(rsp: str, composite_task: CompositeTask):
    rsp = json.loads(rsp)
    tasks = [Task(**task_config) for task_config in rsp]

    if len(tasks) == 1 or tasks[0].dependent_task_ids:
        if tasks[0].dependent_task_ids and len(tasks) > 1:
            # tasks[0].dependent_task_ids means the generated tasks are not a complete plan
            # for they depend on tasks in the current plan, in this case, we only support updating one task each time
            logger.warning(
                "Current plan will take only the first generated task if the generated tasks are not a complete plan"
            )
        # handle a single task
        if composite_task.has_task_id(tasks[0].id):
            # replace an existing task
            composite_task.replace_task(tasks[0])
        else:
            # append one task
            composite_task.append_task(tasks[0])

    else:
        # add tasks in general
        composite_task.add_tasks(tasks)


def precheck_update_plan_from_rsp(rsp: str, composite_task: CompositeTask) -> Tuple[bool, str]:
    original_state = composite_task.__getstate__()  # 保存原始状态
    try:
        update_plan_from_rsp(rsp, composite_task)
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        composite_task.__setstate__(original_state)  # 确保无论是否抛出异常都恢复原始状态
