from abc import ABC, abstractmethod

from thinker_ai.user.user import User


class UserRepository(ABC):
    @abstractmethod
    def save(self, user: User):
        raise NotImplementedError

    @abstractmethod
    def load(self, id: str) -> User:
        raise NotImplementedError
