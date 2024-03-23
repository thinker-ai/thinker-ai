from typing import List, Optional, Any, Dict

from openai.types.beta.assistant_create_params import AssistantToolParam

from thinker_ai.agent.agent import Agent
from thinker_ai.agent.agent_repository import AgentRepository
from thinker_ai.llm import gpt


def create_agent(model: str, user_id: str, name: str, instructions: str = None,
                 tools: List[AssistantToolParam] = None, file_ids: List[str] = None) -> str:
    return AgentRepository.add_agent(model=model,
                                     user_id=user_id,
                                     name=name,
                                     instructions=instructions,
                                     tools=tools,
                                     file_ids=file_ids)


def del_agent(user_id: str, id: str) -> bool:
    return AgentRepository.del_agent(user_id, id)


def get_all_agent_ids(user_id: str) -> List:
    return AgentRepository.get_all_agent_ids(user_id)


def upload_file(file_dir: str) -> str:
    file = gpt.llm.files.create(
        file=open(file_dir, "rb"),
        purpose='assistants'
    )
    return file.id


def delete_file(file_id: str) -> bool:
    deleted = gpt.llm.files.delete(file_id)
    return deleted.deleted


def ask(user_id: str, agent_name: str,topic:str, content: str, file_ids: List[str] = None) -> Dict[str, Any]:
    agent: Agent = AgentRepository.get_agent_by_name(user_id, agent_name)
    result: Dict[str, Any] = {}
    if agent:
        result = agent.ask(topic=topic,content=content, file_ids=file_ids)
    return result
