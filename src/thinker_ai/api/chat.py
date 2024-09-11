from urllib.request import Request

from fastapi import APIRouter, Depends

from thinker_ai.agent.assistant_api_builder import AssistantApiBuilder
from thinker_ai.agent.assistant_api_repository import AssistantRepository
from thinker_ai.agent.provider.llm import LLM
from thinker_ai.agent.openai_assistant_api import openai
from thinker_ai.session_manager import get_session
from fastapi import Request
chat_router = APIRouter()
assistant_repository: AssistantRepository = AssistantRepository.get_instance()


@chat_router.post("/chat", response_model=str)
async def chat(request: Request, session: dict = Depends(get_session)) -> str:
    user_id = session.get("user_id")
    # 获取请求体的 JSON 数据
    data = await request.json()
    topic = data.get("topic")
    content = data.get("content")
    if user_id:
        assistant_name = data.get("assistant_name")
        if assistant_name:
            assistant_id = assistant_repository.get_by_name(assistant_name)
            if not assistant_id:
                assistant_api = AssistantApiBuilder.create(name=assistant_name,
                                                           instructions="你是一个可以根据需要展示不同用户界面的的智能助理，不要"
                                                                        "假设将哪些值插入函数，如果用户的要求模棱两可，请要求说明。")
                assistant_api.load_callables(openai.callables_register.callable_names)
            else:
                assistant_api = AssistantApiBuilder.retrieve(assistant_id)
            result = assistant_api.ask(user_id=user_id, topic_name=topic, content=content)
            return result
        else:
            return await LLM().aask(content)
    else:
        return f"user_id {user_id} not found"
