from fastapi import APIRouter, Depends
from pydantic import BaseModel
from thinker_ai.agent.assistant_api_builder import AssistantApiBuilder
from thinker_ai.agent.assistant_api_repository import AssistantRepository
from thinker_ai.agent.provider.llm import LLM
from thinker_ai.agent.openai_assistant_api import openai
from thinker_ai.api.login import get_session

chat_router = APIRouter()
assistant_repository: AssistantRepository = AssistantRepository.get_instance()


class ChatRequest(BaseModel):
    assistant_name: str
    topic: str
    content: str


@chat_router.post("/chat", response_model=str)
async def chat(request: ChatRequest, session: dict = Depends(get_session)) -> str:
    user_id = session.get("user_id")
    if user_id:
        if request.assistant_name:
            assistant_id = assistant_repository.get_by_name(request.assistant_name)
            if not assistant_id:
                assistant_api = AssistantApiBuilder.create(name=request.assistant_name,
                                                           instructions="你是一个可以根据需要展示不同用户界面的的智能助理，不要"
                                                                        "假设将哪些值插入函数，如果用户的要求模棱两可，请要求说明。")
                assistant_api.load_callables(openai.callables_register.callable_names)
            else:
                assistant_api = AssistantApiBuilder.retrieve(assistant_id)
            result = assistant_api.ask(user_id=user_id, topic_name=request.topic, content=request.content)
            return result
        else:
            return await LLM().aask(request.content)
    else:
        return f"user_id {user_id} not found"
