from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any

from thinker_ai.agent.actions.di.task_desc import TaskType, TaskDesc
from thinker_ai.agent.memory.memory import Memory
from thinker_ai.agent.provider.schema import Message
from thinker_ai.common.logs import logger
from typing import Optional
from pydantic import BaseModel
from thinker_ai.agent.actions import ExecuteNbCode, WriteAnalysisCode
from thinker_ai.agent.actions.di.tool_recommend import ToolRecommender

from typing import List
from thinker_ai.agent.actions import Action


class TaskRepository:
    def __init__(self, file_path: str = None):
        self._file_path = file_path
        self.storage: Dict[str, Task] = {}

    def _save_data_to_file(self):
        if self._file_path is not None:
            Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, 'w', encoding='utf-8') as file:
                json.dump({task_id: task.to_dict() for task_id, task in self.storage.items()},
                          file, ensure_ascii=False, indent=4)

    def _load_data_from_file(self):
        if self._file_path is not None:
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
        self._load_data_from_file()
        return self.storage.get(task_id)

    def delete(self, task_id: str) -> Optional[Task]:
        task = self.storage.pop(task_id, None)
        self._save_data_to_file()
        return task

    def clear(self):
        self.storage.clear()
        self._save_data_to_file()


tasks_storage = TaskRepository()
code_executor: ExecuteNbCode = ExecuteNbCode()
exec_logger: Memory = Memory()


class TaskResult(BaseModel):
    """Result of taking a task, with result and is_success required to be filled"""
    code: str = ""
    result: str
    is_success: bool


class Task(BaseModel):
    id: str
    instruction: str
    type: str = "other"
    dependent_task_ids: Optional[List[str]] = None
    parent_id: Optional[str] = None
    use_reflection: bool = False
    tools: Optional[List[str]] = None
    tool_recommender: Optional[ToolRecommender] = None  # Assuming ToolRecommender is a str for simplicity
    human_confirm: bool = False
    is_finished: bool = False
    task_execute_max_retry: int = 3
    task_result: Optional[TaskResult] = None

    @property
    def task_desc(self) -> TaskDesc:
        map: dict[str, Any] = {
            "parent_id": self.parent_id,
            "id": self.id,
            "dependent_task_ids": self.dependent_task_ids or [],
            "instruction": self.instruction,
            "type": self.type,
        }
        return TaskDesc(map=map)

    def set_tools(self, tools: List[str], tool_recommender: ToolRecommender = None):
        self.tools = tools
        self.tool_recommender = tool_recommender
        if self.tools and not self.tool_recommender:
            self.tool_recommender = ToolRecommender(tools=self.tools)

    async def get_tool_info(self) -> str:
        if self.tool_recommender:
            tool_info = await self.tool_recommender.get_recommended_tool_info_by_bm25(
                parent_task_desc=self.parent_task.task_desc,
                task_desc=self.task_desc)
        else:
            tool_info = ""
        return tool_info

    def reset(self):
        self.task_result = None
        self.is_finished = False

    def to_dict(self):
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'type': self.type,
            'instruction': self.instruction,
            'dependent_task_ids': self.dependent_task_ids,
            'use_reflection': self.use_reflection,
            'human_confirm': self.human_confirm,
            'is_finished': self.is_finished,
            'task_code': self.task_result.code,
            'task_result': self.task_result.result,
        }

    @property
    def parent_task(self) -> Task:
        return tasks_storage.load(self.parent_id) if self.parent_id else ""

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls(
            id=data.get('id'),
            parent_id=data.get('parent_id'),
            type=data.get('type'),
            instruction=data.get('instruction'),
            dependent_task_ids=data.get('dependent_task_ids'),
            use_reflection=bool(data.get('use_reflection', False)),
            human_confirm=bool(data.get('human_confirm', False)))
        is_finished = bool(data.get('is_finished', False))
        if is_finished:
            instance.task_result = TaskResult(is_finished=is_finished, code=data['task_code'],
                                              result=data['task_result'])
        return instance

    async def write_and_exec_code(self,first_trial: bool = True)->bool:
        try:
            code, cause_by = await self._write_code(first_trial)
            exec_logger.add(Message(content=code, role="assistant", cause_by=cause_by))
            result, success = await code_executor.run(code)
            exec_logger.add(Message(content=result, role="user", cause_by=ExecuteNbCode))
            if not success:
                logger.info("coding fail!")
                return False
            review, confirmed = await self._ask_review(human_confirm=False)
        except Exception as e:
            error_msg = f"code fail:{str(e)}"
            logger.error(error_msg)
            exec_logger.add(Message(content=error_msg, role="user", cause_by=ExecuteNbCode))
            return False
        if ReviewConst.CHANGE_WORDS[0] in review:
            return False
        self.is_finished = confirmed
        self.task_result = TaskResult(code=code, result=result, is_success=confirmed)
        exec_logger.clear()
        return True

    async def _ask_review(self, human_confirm: bool = None, review_context_len: int = 5):
        """
        Ask to review the task result, reviewer needs to provide confirmation or request change.
        If human confirms the task result, then we deem the task completed, regardless of whether the code run succeeds;
        if auto mode, then the code run has to succeed for the task to be considered completed.
        """
        exec_logs = exec_logger.get()
        human_confirm = human_confirm or self.human_confirm
        if human_confirm:
            review, confirmed = await AskReview().run(
                exec_logs=exec_logs[-review_context_len:], task_desc=self.task_desc,
                trigger=ReviewConst.CODE_REVIEW_TRIGGER
            )
            if not confirmed:
                exec_logger.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        else:
            ## TODO:需要自动审查
            return "", True

    async def _write_code(
            self,
            first_trial: bool = True
    ):
        todo = WriteAnalysisCode()
        logger.info(f"ready to {todo.name}")
        tool_info = await self.get_tool_info()
        code = await todo.run(
            parent_task_desc=self.parent_task.task_desc if self.parent_task else None,
            plan_status=self.parent_task.plan_status if self.parent_task else None,
            task_desc=self.task_desc,
            tool_info=tool_info,
            exec_logs=exec_logger.get(),
            use_reflection = not first_trial and self.use_reflection
        )

        return code, todo


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
            self,  task_desc: TaskDesc,exec_logs: list[Message] = None,trigger: str = ReviewConst.TASK_REVIEW_TRIGGER
    ) -> Tuple[str, bool]:
        if task_desc:
            logger.info("Current overall plan:")
            logger.info(task_desc.plan)

        logger.info("Most recent context:")
        if exec_logs:
            latest_action = exec_logs[-1].cause_by if exec_logs and exec_logs[-1].cause_by else ""
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
