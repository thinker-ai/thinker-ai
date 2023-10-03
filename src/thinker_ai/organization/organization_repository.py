from abc import ABC, abstractmethod

from thinker_ai.organization.organization import Organization


class OrganizationRepository(ABC):
    @abstractmethod
    def save(self, organization: Organization):
        raise NotImplementedError

    @abstractmethod
    def load(self, name: str) -> Organization:
        raise NotImplementedError
