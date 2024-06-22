from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict

from thinker_ai.agent.memory.memory import Memory
from thinker_ai.agent.provider.schema import Message
from thinker_ai.common.logs import logger
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from thinker_ai.agent.actions import ExecuteNbCode, WriteAnalysisCode
from thinker_ai.agent.actions.di.tool_recommend import BM25ToolRecommender, ToolRecommender


from typing import List
from thinker_ai.agent.actions import Action
from thinker_ai.common.val_class import remove_comments
from thinker_ai.utils.code_parser import CodeParser

from thinker_ai.agent.prompts.task_type import (
    DATA_PREPROCESS_PROMPT,
    EDA_PROMPT,
    FEATURE_ENGINEERING_PROMPT,
    IMAGE2WEBPAGE_PROMPT,
    MODEL_EVALUATE_PROMPT,
    MODEL_TRAIN_PROMPT,
)
from thinker_ai.configs.const import DATA_PATH


class TaskRepository:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self.storage: Dict[str, Task] = {}
        self._load_data_from_file()

    def _save_data_to_file(self):
        Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, 'w', encoding='utf-8') as file:
            json.dump({task_id: task.to_dict() for task_id, task in self.storage.items()},
                      file, ensure_ascii=False, indent=4)

    def _load_data_from_file(self):
        try:
            with open(self._file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.storage = {task_id: Task.from_dict(task_data) for task_id, task_data in data.items()}
        except (IOError, json.JSONDecodeError):
            self.storage = {}

    def save(self, task: Task):
        self.storage[task.id] = task
        self._save_data_to_file()

    def load(self, task_id: str) -> Optional[Task]:
        return self.storage.get(task_id)

    def delete(self, task_id: str) -> Optional[Task]:
        task = self.storage.pop(task_id, None)
        self._save_data_to_file()
        return task

    def clear(self):
        self.storage.clear()
        self._save_data_to_file()


tasks_storage = TaskRepository(DATA_PATH / "tasks.json")
code_executor: ExecuteNbCode = ExecuteNbCode()
exec_logger: Memory = Memory()
class TaskTypeDef(BaseModel):
    name: str
    desc: str = ""
    guidance: str = ""


class TaskType(Enum):
    """By identifying specific types of tasks, we can inject human priors (guidance) to help task solving"""

    EDA = TaskTypeDef(
        name="eda",
        desc="For performing exploratory data analysis",
        guidance=EDA_PROMPT,
    )
    DATA_PREPROCESS = TaskTypeDef(
        name="data preprocessing",
        desc="For preprocessing dataset in a data analysis or machine learning task ONLY,"
             "general data operation doesn't fall into this type",
        guidance=DATA_PREPROCESS_PROMPT,
    )
    FEATURE_ENGINEERING = TaskTypeDef(
        name="feature engineering",
        desc="Only for creating new columns for input data.",
        guidance=FEATURE_ENGINEERING_PROMPT,
    )
    MODEL_TRAIN = TaskTypeDef(
        name="model train",
        desc="Only for training model.",
        guidance=MODEL_TRAIN_PROMPT,
    )
    MODEL_EVALUATE = TaskTypeDef(
        name="model evaluate",
        desc="Only for evaluating model.",
        guidance=MODEL_EVALUATE_PROMPT,
    )
    IMAGE2WEBPAGE = TaskTypeDef(
        name="image2webpage",
        desc="For converting image into webpage code.",
        guidance=IMAGE2WEBPAGE_PROMPT,
    )
    OTHER = TaskTypeDef(name="other", desc="Any tasks not in the defined categories")

    # Legacy TaskType to support tool recommendation using type match. You don't need to define task types if you have no human priors to inject.
    TEXT2IMAGE = TaskTypeDef(
        name="text2image",
        desc="Related to text2image, image2image using stable diffusion model.",
    )
    WEBSCRAPING = TaskTypeDef(
        name="web scraping",
        desc="For scraping data from web pages.",
    )
    EMAIL_LOGIN = TaskTypeDef(
        name="email login",
        desc="For logging to an email.",
    )

    @property
    def type_name(self):
        return self.value.name

    @classmethod
    def get_type(cls, type_name: str) -> Optional[TaskTypeDef]:
        for member in cls:
            if member.type_name == type_name:
                return member.value
        return None


class TaskResult(BaseModel):
    """Result of taking a task, with result and is_success required to be filled"""
    code: str = ""
    result: str
    is_success: bool


class Task(BaseModel):
    STRUCTURAL_CONTEXT: str = """
    ## User Requirement
    {user_requirement}
    ## Context
    {context}
    ## Current Task
    {current_task}
    """
    id: str
    instruction: str
    task_type: str = "OTHER"
    dependent_task_ids: Optional[List[str]] = None
    parent_id: Optional[str] = None
    use_reflection: bool = False
    tools: Optional[List[str]] = None
    tool_recommender: Optional[ToolRecommender] = None  # Assuming ToolRecommender is a str for simplicity
    auto_run: bool = True
    is_finished: bool = False
    context: str = ""
    task_result: Optional[TaskResult] = None


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_recommender = BM25ToolRecommender(tools=self.tools if self.tools else []) \
            if self.tool_recommender is None else self.tool_recommender

    @property
    async def tool_info(self) -> str:
        if self.tool_recommender:
            context = (
                exec_logger.get()[-1].content if exec_logger.get() else ""
            )  # thoughts from _think stage in 'react' mode
            plan = self.planner.plan if self.use_plan else None
            tool_info = await self.tool_recommender.get_recommended_tool_info(context=context,
                                                                              task_instruction=self.parent_task.instruction)
        else:
            tool_info = ""
        return tool_info

    def reset(self):
        self.task_result = None
        self.is_finished = False

    def set_task_result(self, task_result: TaskResult):
        self.task_result = task_result
        self.is_finished = True

    def to_dict(self):
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'type': self.task_type,
            'instruction': self.instruction,
            'dependent_task_ids': self.dependent_task_ids,
            'use_reflection': self.use_reflection,
            'auto_run': self.auto_run
        }

    @property
    def parent_task(self) -> Task:
        return tasks_storage.load(self.parent_id) if self.parent_id else ""

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data['id'],
            parent_id=data['parent_id'],
            task_type=data['type'],
            instruction=data['instruction'],
            dependent_task_ids=data['dependent_task_ids'],
            use_reflection=data['use_reflection'],
            auto_run=data['auto_run'])

    async def act(self) -> TaskResult:
        """Useful in 'plan_and_act' mode. Wrap the output in a TaskResult for review and confirmation."""
        await self._write_and_exec_code()
        return self.task_result

    async def _write_and_exec_code(self, max_retry: int = 3):
        counter = 0

        while True:

            ### write code ###
            code, cause_by = await self._write_code(counter)

            exec_logger.add(Message(content=code, role="assistant", cause_by=cause_by))

            ### execute code ###
            result, success = await code_executor.run(code)
            print(result)

            exec_logger.add(Message(content=result, role="user", cause_by=ExecuteNbCode))

            ### process execution result ###
            counter += 1

            if not success and counter >= max_retry:
                logger.info("coding failed!")
                review, _ = await self.ask_review(auto_run=False)
                if ReviewConst.CHANGE_WORDS[0] in review:
                    counter = 0  # redo the task again with help of human suggestions
            else:
                self.task_result = TaskResult(code=code, result=result, is_success=success)
                break

    def get_context_msg(self, task_exclude_field=None) -> list[Message]:
        """only to reduce context length and improve performance"""
        context = self.STRUCTURAL_CONTEXT.format(
            user_requirement=self.instruction, context=self.context, current_task=self.model_dump_json()
        )
        context_msg = [Message(content=context, role="user")]
        return context_msg + exec_logger.get()

    async def _write_code(
            self,
            counter: int
    ):
        todo = WriteAnalysisCode()
        logger.info(f"ready to {todo.name}")
        use_reflection = counter > 0 and self.use_reflection  # only use reflection after the first trial
        parent_status = self.parent_task.plane_status_msg if self.parent_task else ""
        code = await todo.run(
            user_requirement=self.instruction,
            plan_status=parent_status,
            tool_info=await self.tool_info,
            working_memory=exec_logger.get(),
            use_reflection=use_reflection,
        )

        return code, todo

    async def ask_review(
            self,
            auto_run: bool = None,
            review_context_len: int = 5,
    ):
        """
        Ask to review the task result, reviewer needs to provide confirmation or request change.
        If human confirms the task result, then we deem the task completed, regardless of whether the code run succeeds;
        if auto mode, then the code run has to succeed for the task to be considered completed.
        """
        auto_run = auto_run or self.auto_run
        if not auto_run:
            context = self.get_context_msg()
            review, confirmed = await AskReview().run(
                context=context[-review_context_len:], trigger=ReviewConst.CODE_REVIEW_TRIGGER
            )
            if not confirmed:
                exec_logger.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        confirmed = self.task_result.is_success if self.task_result else True
        return "", confirmed


class ReviewConst:
    TASK_REVIEW_TRIGGER = "task"
    CODE_REVIEW_TRIGGER = "code"
    CONTINUE_WORDS = ["confirm", "continue", "c", "yes", "y"]
    CHANGE_WORDS = ["change"]
    EXIT_WORDS = ["exit"]
    TASK_REVIEW_INSTRUCTION = (
        f"If you want to change, add, delete a task or merge tasks in the plan, say '{CHANGE_WORDS[0]} task task_id or current task, ... (things to change)' "
        f"If you confirm the output from the current task and wish to continue, type: {CONTINUE_WORDS[0]}"
    )
    CODE_REVIEW_INSTRUCTION = (
        f"If you want the codes to be rewritten, say '{CHANGE_WORDS[0]} ... (your change advice)' "
        f"If you want to leave it as is, type: {CONTINUE_WORDS[0]} or {CONTINUE_WORDS[1]}"
    )
    EXIT_INSTRUCTION = f"If you want to terminate the process, type: {EXIT_WORDS[0]}"


class AskReview(Action):
    async def run(
            self, context: list[Message] = [], composite_task: Task = None,
            trigger: str = ReviewConst.TASK_REVIEW_TRIGGER
    ) -> Tuple[str, bool]:
        if composite_task:
            logger.info("Current overall plan:")
            logger.info(
                "\n".join(
                    [f"{task.id}: {task.instruction}, is_finished: {task.is_finished}"
                     for task in composite_task.get_finished_tasks()]
                )
            )

        logger.info("Most recent context:")
        latest_action = context[-1].cause_by if context and context[-1].cause_by else ""
        review_instruction = (
            ReviewConst.TASK_REVIEW_INSTRUCTION
            if trigger == ReviewConst.TASK_REVIEW_TRIGGER
            else ReviewConst.CODE_REVIEW_INSTRUCTION
        )
        prompt = (
            f"This is a <{trigger}> review. Please review output from {latest_action}\n"
            f"{review_instruction}\n"
            f"{ReviewConst.EXIT_INSTRUCTION}\n"
            "Please type your review below:\n"
        )

        rsp = input(prompt)

        if rsp.lower() in ReviewConst.EXIT_WORDS:
            exit()

        # Confirmation can be one of "confirm", "continue", "c", "yes", "y" exactly, or sentences containing "confirm".
        # One could say "confirm this task, but change the next task to ..."
        confirmed = rsp.lower() in ReviewConst.CONTINUE_WORDS or ReviewConst.CONTINUE_WORDS[0] in rsp.lower()

        return rsp, confirmed


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


class CheckData(Action):
    async def run(self, finished_tasks: List[Task]) -> str:
        code_written = [remove_comments(task.code) for task in finished_tasks]
        code_written = "\n\n".join(code_written)
        prompt = CHECK_DATA_PROMPT.format(code_written=code_written)
        rsp = await self._aask(prompt)
        code = CodeParser.parse_code(block=None, text=rsp)
        return code
