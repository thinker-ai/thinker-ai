from abc import ABC, abstractmethod
from typing import List,Dict


class AssistantApi(ABC):

    @property
    @abstractmethod
    def id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the assistant."""
        raise NotImplementedError

    @name.setter
    @abstractmethod
    def name(self, new_value):
        raise NotImplementedError

    @abstractmethod
    def tools(self) -> List[Dict]:
        """Returns the list of tools enabled on the assistant."""

    @property
    @abstractmethod
    def file_ids(self) -> List[str]:
        """Returns the list of file IDs used by the assistant's tools."""

    @abstractmethod
    def set_instructions(self, instructions: str) -> None:
        """Sets the system instructions for the assistant."""

    @abstractmethod
    def register_file_id(self, file_id: str) -> None:
        """Registers a file ID for the code interpreter tool."""

    @abstractmethod
    def register_vector_store_id(self, vector_store_id: str) -> None:
        """Registers a vector store ID for the file search tool."""

    @abstractmethod
    def remove_file_id(self, file_id: str) -> None:
        """Removes a file ID from the code interpreter tool."""

    @abstractmethod
    def remove_vector_store_id(self, vector_store_id: str) -> None:
        """Removes a vector store ID from the file search tool."""

    @abstractmethod
    def ask(self,user_id:str, content: str, topic_name: str = "default") -> str:
        """Sends a query to the assistant and returns the response."""

    @abstractmethod
    def register_code_interpreter(self) -> None:
        """Registers the code interpreter tool."""

    @abstractmethod
    def remove_code_interpreter(self) -> None:
        """Removes the code interpreter tool."""

    @abstractmethod
    def register_file_search(self) -> None:
        """Registers the file search tool."""

    @abstractmethod
    def remove_file_search(self) -> None:
        """Removes the file search tool."""

    def load_callables(self, callable_names:list[str]):
        """add callable."""

    def unload_callables(self, callable_names:list[str]):
        """remove callable"""

    @abstractmethod
    def unload_all_callables(self) -> None:
        """Removes all registered callables."""

    @abstractmethod
    def is_function_in_assistant(self, name: str) -> bool:
        """Checks if a function is registered."""

    @staticmethod
    @abstractmethod
    def get_most_similar_strings(source_strings: List[str], compare_string: str, k: int = 1,
                                 embedding_model: str = "text-embedding-3-small") -> List[tuple[str, float]]:
        """Returns the most similar strings from the source strings compared to the provided string."""

    @abstractmethod
    def get_most_similar_from_file(self, file_id: str, compare_string: str, k: int = 1,
                                   embedding_model: str = "text-embedding-3-small") -> List[tuple[str, float]]:
        """Returns the most similar strings from a file compared to the provided string."""
