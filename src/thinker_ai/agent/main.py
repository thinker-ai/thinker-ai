from typing import List, Optional, Any, Dict
from openai.types.beta.assistant_create_params import AssistantToolParam

from thinker_ai.agent.agent import Agent
from thinker_ai.agent.agent_dao import AgentDAO
from thinker_ai.agent.agent_repository import AgentRepository
from thinker_ai.agent.llm import gpt

agent_repository = AgentRepository.get_instance()


def create_agent(model: str, user_id: str, name: str, instructions: str = None, description: str = None,
                 tools: List[AssistantToolParam] = None, file_ids: List[str] = None) -> str:
    assistant = gpt.llm.beta.assistants.create(
        name=name,
        model=model,
        instructions=instructions,
        tools=tools,
        file_ids=file_ids,
        description=description)
    agent = Agent(id=assistant.id, user_id=user_id, assistant=assistant, threads={}, client=gpt.llm)
    return agent_repository.add_agent(agent=agent, user_id=user_id)


def del_agent(user_id: str, id: str) -> bool:
    return agent_repository.delete_agent(user_id, id)


def get_all_agent_ids(user_id: str) -> List:
    return agent_repository.get_my_agent_ids(user_id)


def upload_file(file_dir: str) -> str:
    return gpt.upload_file(file_dir).id


def delete_file(file_id: str) -> bool:
    return gpt.delete_file(file_id)

def ask(user_id: str, agent_name: str, topic: str, content: str) -> List[Dict[str, Any]]:
    agent: Agent = agent_repository.get_agent(user_id, agent_name)
    result: List[Dict[str, Any]] = []
    if agent:
        result = agent.ask(topic=topic, content=content)
    return result
