from typing import List, cast, Literal
from openai.types.beta.assistant_create_params import AssistantToolParam

from thinker_ai.agent.openai_assistant.openai_assistant import OpenAiAssistant
from thinker_ai.agent.openai_assistant.openai_assistant_repository import OpenAiAssistantRepository
from thinker_ai.agent.provider import OpenAILLM
from thinker_ai.context_mixin import ContextMixin

agent_repository = OpenAiAssistantRepository.get_instance()
open_ai = (cast(OpenAILLM, ContextMixin().llm))


def create_agent(model: str, user_id: str, name: str, instructions: str = None, description: str = None,
                 tools: List[AssistantToolParam] = None, file_ids: List[str] = None) -> str:
    assistant = open_ai.client.beta.assistants.create(
        name=name,
        model=model,
        instructions=instructions,
        tools=tools,
        file_ids=file_ids,
        description=description)
    agent = OpenAiAssistant.from_instance(user_id, assistant)
    return agent_repository.add_assistant(assistant=agent, user_id=user_id)


def del_agent(user_id: str, assistant_id: str) -> bool:
    return agent_repository.delete_assistant(user_id, assistant_id)


def get_all_agent_ids(user_id: str) -> List:
    return agent_repository.get_my_assistant_ids(user_id)


def upload_file(file_dir: str, purpose: Literal["fine-tune", "assistants"] = "assistants") -> str:
    return open_ai.upload_file(file_dir, purpose).id


def delete_file(file_id: str) -> bool:
    return open_ai.delete_file(file_id)


def ask(user_id: str, agent_name: str, topic: str, content: str) -> str:
    agent: OpenAiAssistant = agent_repository.get_assistant(user_id, agent_name)
    result = ""
    if agent:
        result = agent.ask(topic=topic, content=content)
    return result
