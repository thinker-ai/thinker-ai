from abc import ABC, abstractmethod
from typing import Any

from thinker_ai.context import Context


class Action(ABC):
    def __init__(self, context: Context):
        self.context = context

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    async def run(self, *args, **kwargs):
        """The run method should be implemented in a subclass"""


class DoAction(ABC):
    def __init__(self, context: Context):
        self.context = context

    @abstractmethod
    async def run(self, previous_review_results: Any = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError


class ReviewAction(ABC):
    def __init__(self, context: Context):
        self.context = context

    @abstractmethod
    async def run(self, do_results: Any = None, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError


class AcceptAction(ABC):
    def __init__(self, context: Context):
        self.context = context

    @abstractmethod
    async def run(self, do_results: Any, review_results: Any = None, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_accept(self) -> bool:
        raise NotImplementedError


class ThinkerAction(Action, ABC):
    def __init__(self, do: DoAction, review: ReviewAction, accept: AcceptAction, context: Context, max_try: int = 3):
        super().__init__(context)
        self.do = do
        self.review = review
        self.accept = accept
        self.max_try = max_try

    async def run(self, *args, **kwargs):
        try_times = 1
        while self.accept or try_times > self.max_try:
            await self.do.run(*args, **kwargs)
            await self.review.run(self.do.get_result(), *args, **kwargs)
            await self.accept.run(self.do.get_result(), self.review.get_result() * args, **kwargs)
            try_times += 1

    def get_result(self) -> Any:
        return self.accept.get_result()

    @abstractmethod
    def is_success(self) -> bool:
        return self.accept.is_accept()