from abc import abstractmethod
from typing import Any, Dict

from thinker_ai.common.serializable import Serializable


class Task(Serializable):
    done:bool = False

    def __init__(self, solution_id:str,id:str,name: str, guide: str, checklist: Dict, goals: str, max_try: int = 3, **data: Any):
        super().__init__(**data)
        self.solution_id: str=solution_id
        self.id: str = id
        self.name = name
        self.guide = guide
        self.checklist = checklist
        self.goals = goals
        self.max_try = max_try

    @abstractmethod
    async def _propose(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    async def _review(self, propose_result: str, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    async def _accept(self, propose_result: str, review_result: str, *args, **kwargs) -> bool:
        raise NotImplementedError

    async def do(self, *args, **kwargs):
        try_times = 1
        while not self.done or try_times > self.max_try:
            propose_result = await self._propose(*args, **kwargs)
            review_result = await self._review(propose_result, *args, **kwargs)
            self.done = await self._accept(propose_result, review_result, *args, **kwargs)
            try_times += 1
