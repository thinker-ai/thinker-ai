import json
from threading import Lock
from typing import List, Optional

from thinker_ai.context import get_project_root
from thinker_ai.utils.serializable import Serializable
from thinker_ai.utils.singleton_meta import SingletonMeta


class ThreadPO(Serializable):
    name: str
    thread_id: str


class AgentPO(Serializable):
    id: str
    user_id: str
    threads: List[ThreadPO] = []
    assistant_id: str


class AgentDAO(metaclass=SingletonMeta):
    _instance = None

    def __init__(self, filepath: str):
        # This ensures that initialization happens only once
        if not AgentDAO._instance:
            self.filepath = filepath or get_project_root() / 'data/agents.json'
            self._lock = Lock()
            self._load_agents()

    @classmethod
    def get_instance(cls, filepath: Optional[str] = get_project_root() / 'data/agents.json') -> "AgentDAO":
        """
        The factory method for getting the singleton instance.
        """
        if not cls._instance:
            cls._instance = cls(filepath)
        return cls._instance

    def _load_agents(self):
        try:
            with open(self.filepath, 'r') as file:
                self.agents = json.load(file)
        except FileNotFoundError:
            self.agents = []

    def _save(self):
        with open(self.filepath, 'w') as file:
            json.dump(self.agents, file, ensure_ascii=False, indent=4)

    def add_agent(self, agent_po):
        with self._lock:
            self.agents.append(agent_po.dict())
            self._save()

    def get_agent(self, agent_id) -> Optional[AgentPO]:
        with self._lock:
            for agent_dict in self.agents:
                if agent_dict['id'] == agent_id:
                    return AgentPO(**agent_dict)
        return None

    def update_agent(self, updated_agent):
        with self._lock:
            for index, agent_dict in enumerate(self.agents):
                if agent_dict['id'] == updated_agent.id:
                    self.agents[index] = updated_agent.dict()
                    self._save()
                    return
            raise ValueError(f"Agent with id {updated_agent.id} not found")

    def delete_agent(self, agent_id):
        with self._lock:
            self.agents = [agent for agent in self.agents if agent['id'] != agent_id]
            self._save()

    def get_my_agent_ids(self, user_id) -> List[str]:
        with self._lock:
            result: List[str] = []
            for agent_dict in self.agents:
                if agent_dict['user_id'] == user_id:
                    result.append(agent_dict['id'])
            return result
