from abc import ABC, abstractmethod
from typing import Any, Dict

from thinker_ai.context import Context


class Skill(ABC):
    def __init__(self,name:str):
        self.name = name

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    async def act(self, *args, **kwargs):
        raise NotImplementedError


class ProposeSkill(Skill,ABC):
    def __init__(self):
        super().__init__("propose")

    @abstractmethod
    async def act(self, previous_review_results: Any = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError


class ReviewSkill(Skill,ABC):
    def __init__(self):
        super().__init__("review")

    @abstractmethod
    async def act(self, propose_results: Any = None, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError


class AcceptSkill(Skill,ABC):
    def __init__(self):
        super().__init__("accept")
    @abstractmethod
    async def act(self, propose_results: Any, review_results: Any = None, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_accept(self) -> bool:
        raise NotImplementedError


class ThinkSkill(Skill, ABC):

    def __init__(self, propose: ProposeSkill, review: ReviewSkill, accept: AcceptSkill, name: str, max_try: int = 3):
        super().__init__("think")
        self.propose = propose
        self.review = review
        self.accept = accept
        self.max_try = max_try

    async def act(self, *args, **kwargs):
        try_times = 1
        while self.accept or try_times > self.max_try:
            await self.propose.act(*args, **kwargs)
            await self.review.act(self.propose.get_result(), *args, **kwargs)
            await self.accept.act(self.propose.get_result(), self.review.get_result() * args, **kwargs)
            try_times += 1

    def get_result(self) -> Any:
        return self.accept.get_result()

    @abstractmethod
    def is_success(self) -> bool:
        return self.accept.is_accept()