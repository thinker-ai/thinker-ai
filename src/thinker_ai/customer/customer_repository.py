from abc import ABC, abstractmethod

from thinker_ai.customer.customer import Customer


class CustomerRepository(ABC):
    @abstractmethod
    def save(self, customer: Customer):
        raise NotImplementedError

    @abstractmethod
    def load(self, name: str) -> Customer:
        raise NotImplementedError