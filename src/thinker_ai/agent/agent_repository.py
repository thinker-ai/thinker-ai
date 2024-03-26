from openai import OpenAI
from openai.types.beta.assistant_create_params import AssistantToolParam

from thinker_ai.agent.agent import Agent

import threading
from typing import Dict, Optional, List

from thinker_ai.agent.llm import gpt


class AgentRepository:
    repository: Dict[str, Dict[str, Agent]] = {}
    locks: Dict[str, threading.Lock] = {}  # 不同用户彼此互不干扰，不会有并发问题，相同用户的不同线程才会有并发问题
    client: OpenAI = gpt.llm

    @classmethod
    def get_lock(cls, user_id: str) -> threading.Lock:
        if user_id not in cls.locks:
            cls.locks[user_id] = threading.Lock()
        return cls.locks[user_id]

    @classmethod
    def get_agent(cls, user_id: str, agent_id: str) -> Optional[Agent]:
        with cls.get_lock(user_id):
            my_agents: Dict[str, Agent] = cls.repository.get(user_id)
            if my_agents is not None:
                return my_agents.get(agent_id)
            else:
                return None

    @classmethod
    def add_agent(cls, model: str, user_id: str, name: str, description: str = None, instructions: str = None,
                  tools: List[AssistantToolParam] = None, file_ids: List[str] = None) -> str:
        with cls.get_lock(user_id):
            my_agents: Dict[str, Agent] = cls.repository.get(user_id)
            if my_agents is None:
                my_agents = {}
                cls.repository[user_id] = my_agents
            assistant = cls.client.beta.assistants.create(
                name=name,
                description=description,
                instructions=instructions,
                tools=tools,
                file_ids=file_ids,
                model=model
            )
            agent = Agent(
                user_id=user_id,
                assistant=assistant
            )
            my_agents[agent.id] = agent
            return agent.id

    @classmethod
    def del_agent(cls, user_id, agent_id: str) -> bool:
        with cls.get_lock(user_id):
            my_agents: Dict[str, Agent] = cls.repository.get(user_id)
            if my_agents is None:
                return False
            my_agents.pop(agent_id)
            deleted = cls.client.beta.assistants.delete(agent_id)
            return deleted.deleted

    @classmethod
    def get_all_agent_ids(cls, user_id) -> List:
        with cls.get_lock(user_id):
            my_agents: Dict[str, Agent] = cls.repository.get(user_id)
            if my_agents is None:
                return []
            return list(my_agents.keys())

    @classmethod
    def get_agent_by_id(cls, user_id, agent_id: str) -> Optional[Agent]:
        with cls.get_lock(user_id):
            my_agents: Dict[str, Agent] = cls.repository.get(user_id)
            if my_agents is None:
                return None
            return my_agents.get(agent_id)

    @classmethod
    def get_agent_by_name(cls, user_id, agent_name: str) -> Optional[Agent]:
        with cls.get_lock(user_id):
            my_agents: Dict[str, Agent] = cls.repository.get(user_id)
            if my_agents is None:
                return None
            for agent in my_agents.values():
                if agent.name == agent_name:
                    return agent
            return None
