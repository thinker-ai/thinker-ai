from threading import Lock
from typing import Optional

from openai import OpenAI

from thinker_ai.agent.agent import Agent
from thinker_ai.agent.agent_dao import AgentDAO, ThreadPO, AgentPO
from thinker_ai.agent.llm import gpt
from thinker_ai.utils.singleton_meta import SingletonMeta


class AgentRepository(metaclass=SingletonMeta):
    def __init__(self, agent_dao: AgentDAO, client: OpenAI):
        self.agent_dao = agent_dao
        self.client = client
        self._lock = Lock()

    def _agent_to_po(self, agent: Agent) -> AgentPO:
        threads_po = [ThreadPO(name=thread.name, thread_id=thread.id) for thread in agent.threads.values()]
        return AgentPO(id=agent.id, user_id=agent.user_id, threads=threads_po, assistant_id=agent.assistant.id)

    def _po_to_agent(self, agent_po: AgentPO) -> Agent:
        assistant = gpt.llm.beta.assistants.retrieve(agent_po.assistant_id)
        threads = {thread_po.thread_id: gpt.llm.beta.threads.retrieve(thread_po.thread_id) for thread_po in
                   agent_po.threads}
        return Agent(id=agent_po.id, user_id=agent_po.user_id, assistant=assistant, threads=threads, client=self.client)

    def add_agent(self, agent: Agent, user_id: str):
        if agent.user_id != user_id:
            raise PermissionError("Cannot add an agent for another user.")
        with self._lock:
            self.agent_dao.add_agent(self._agent_to_po(agent))

    def get_agent(self, agent_id: str, user_id: str) -> Optional[Agent]:
        with self._lock:
            agent_po = self.agent_dao.get_agent(agent_id)
            if agent_po is not None and agent_po.user_id == user_id:
                return self._po_to_agent(agent_po)
            return None

    def update_agent(self, agent: Agent, user_id: str):
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
