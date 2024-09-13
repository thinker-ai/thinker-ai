from urllib.request import Request

from fastapi import APIRouter, Depends

from thinker_ai.agent.assistant_api_builder import AssistantApiBuilder
from thinker_ai.agent.assistant_api_repository import AssistantRepository
from thinker_ai.agent.provider.llm import LLM
from thinker_ai.agent.openai_assistant_api import openai
from thinker_ai.session_manager import get_session
from fastapi import Request
from textwrap import dedent

chat_router = APIRouter()
assistant_repository: AssistantRepository = AssistantRepository.get_instance()
instructions = """
You are an intelligent assistant that can answer user questions and give use tools 
at the same time; you can respond to humans only, or use command tools only, or 
perform both at the same time
"""
description = """
Please answer questions or use tools based on the user's input, You are now at a certain step in the whole execution session, 
here is the previous chat and tools actions of your conversation:
'''
{{previous_chat_and_tools_actions}}
'''
refer carefully to the previous chat and tools actions and consider what you can do next, if you are using the tool, 
do not make assumptions about which values to insert into the next function, if ambiguous, ask the user for assistance
 or, alternatively, use other tools to make this information explicit, e.g., to view and manipulate the user's screen, 
 etc., and you can keep on sustaining this behavior until you obtain the your desired result.
"""


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
                                                           instructions=instructions,
                                                           description=description)
                assistant_api.load_callables(openai.callables_register.callable_names)
            else:
                assistant_api = AssistantApiBuilder.retrieve(assistant_id)
            result = assistant_api.ask(user_id=user_id, topic_name=topic, content=content)
            return result
        else:
            return await LLM().aask(content)
    else:
        return f"user_id {user_id} not found"
