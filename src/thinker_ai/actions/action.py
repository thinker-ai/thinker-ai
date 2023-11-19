from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel


class BaseAction(ABC):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    async def execute(self, *args, **kwargs):
        raise NotImplementedError


class Criteria(BaseModel):
    name: str
    guide: str
    checklist: Dict
    goals: str


class ProposeAction(BaseAction, ABC):
    def __init__(self, criteria: Criteria):
        super().__init__("propose")
        self.criteria = criteria

    @abstractmethod
    async def execute(self, msg: str, previous_review_results: Any = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError


class ReviewAction(BaseAction, ABC):
    def __init__(self, criteria: Criteria):
        super().__init__("review")
        self.criteria = criteria

    @abstractmethod
    async def execute(self, propose_results: Any = None, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError


class AcceptAction(BaseAction, ABC):
    def __init__(self, criteria: Criteria):
        super().__init__("accept")
        self.criteria = criteria

    @abstractmethod
    async def execute(self, propose_results: Any, review_results: Any = None, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_accept(self) -> bool:
        raise NotImplementedError


class ProposeReviewAcceptAction(BaseAction, ABC):

    def __init__(self, criteria: Criteria, max_try: int = 3):
        super().__init__("think")
        self.propose = ProposeAction(criteria)
        self.review = ReviewAction(criteria)
        self.accept = AcceptAction(criteria)
        self.max_try = max_try

    async def execute(self, *args, **kwargs):
        try_times = 1
        while self.accept.is_accept() or try_times > self.max_try:
            await self.propose.execute(*args, **kwargs)
            await self.review.execute(self.propose.get_result(), *args, **kwargs)
            await self.accept.execute(self.propose.get_result(), self.review.get_result() * args, **kwargs)
            try_times += 1

    def get_result(self) -> Any:
        return self.accept.get_result()

    @abstractmethod
    def is_success(self) -> bool:
        return self.accept.is_accept()
