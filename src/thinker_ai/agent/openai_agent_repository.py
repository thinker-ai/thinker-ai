from threading import Lock
from typing import Optional, List, cast

from thinker_ai.agent.openai_assistant_agent import AssistantAgent
from thinker_ai.agent.agent_dao import AgentDAO, ThreadPO, AgentPO
from thinker_ai.agent.provider import OpenAILLM
from thinker_ai.configs.const import PROJECT_ROOT
from thinker_ai.common.singleton_meta import SingletonMeta
from thinker_ai.context_mixin import ContextMixin


class AgentRepository(metaclass=SingletonMeta):
    _instance = None

    @classmethod
    def get_instance(cls, filepath: Optional[str] = PROJECT_ROOT/ 'data/agents.q') -> "AgentRepository":
        if not cls._instance:
            cls._instance = cls(filepath)
        return cls._instance

    def __init__(self, filepath: str):
        if not AgentRepository._instance:
            self.agent_dao = AgentDAO.get_instance(filepath)
            self.client = (cast(OpenAILLM, ContextMixin().llm)).client
            self._lock = Lock()
            AgentRepository._instance = self
        else:
            raise Exception("Attempting to instantiate a singleton class.")

    def _agent_to_po(self, agent: AssistantAgent) -> AgentPO:
        threads_po = [ThreadPO(name=thread.name, thread_id=thread.id) for thread in agent.threads.values()]
        return AgentPO(id=agent.id, user_id=agent.user_id, threads=threads_po, assistant_id=agent.assistant.id)

    def _po_to_agent(self, agent_po: AgentPO) -> AssistantAgent:
        assistant = self.client.beta.assistants.retrieve(agent_po.assistant_id)
        threads = {thread_po.thread_id:self.client.beta.threads.retrieve(thread_po.thread_id) for thread_po in
                   agent_po.threads}
        return AssistantAgent(id=agent_po.id, user_id=agent_po.user_id, assistant=assistant, threads=threads, client=self.client)

    def add_agent(self, agent: AssistantAgent, user_id: str):
        if agent.user_id != user_id:
            raise PermissionError("Cannot add an agent for another user.")
        with self._lock:
            self.agent_dao.add_agent(self._agent_to_po(agent))

    def get_agent(self, agent_id: str, user_id: str) -> Optional[AssistantAgent]:
        with self._lock:
            agent_po = self.agent_dao.get_agent(agent_id)
            if agent_po is not None and agent_po.user_id == user_id:
                return self._po_to_agent(agent_po)
            return None

    def get_my_agent_ids(self, user_id) -> List[str]:
        with self._lock:
            return self.agent_dao.get_my_agent_ids(user_id)

    def update_agent(self, agent: AssistantAgent, user_id: str):
        if agent.user_id != user_id:
            raise PermissionError("Cannot update an agent for another user.")
        with self._lock:
            self.agent_dao.update_agent(self._agent_to_po(agent))

    def delete_agent(self, agent_id: str, user_id: str):
        agent = self.get_agent(agent_id, user_id)
        if agent is None:
            raise PermissionError("Cannot delete an agent for another user.")
        with self._lock:
            self.agent_dao.delete_agent(agent_id)

    @classmethod
    def reset_instance(cls):
        cls._instance = None
