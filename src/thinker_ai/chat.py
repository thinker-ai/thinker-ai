import os

from fastapi import APIRouter, Depends
from fastapi import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.templating import Jinja2Templates

from thinker_ai.agent.assistant_api_builder import AssistantApiBuilder
from thinker_ai.agent.assistant_api_repository import AssistantRepository
from thinker_ai.agent.provider.llm import LLM
from thinker_ai.login import get_session
from thinker_ai.configs.const import PROJECT_ROOT
from thinker_ai.agent.openai_assistant_api import openai

chat_router = APIRouter()
root_dir = PROJECT_ROOT
template_dir = os.path.join(root_dir, 'web', 'templates')
templates = Jinja2Templates(directory=template_dir)
assistant_repository:AssistantRepository = AssistantRepository.get_instance()
# 添加调试信息
print(f"Template directory: {template_dir}")


class ChatRequest(BaseModel):
    assistant_name: str
    topic: str
    content: str


@chat_router.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@chat_router.get("/mermaid", response_class=HTMLResponse)
async def mermaid(request: Request):
    return templates.TemplateResponse("mermaid.html", {"request": request})


@chat_router.post("/chat", response_model=str)
async def chat(request: ChatRequest, session: dict = Depends(get_session)) -> str:
    user_id = session.get("user_id")
    if user_id:
        if request.assistant_name:
            # name=openai.callables_register.register_callable(get_service_loader().load_ui_and_show, LoadArgs)
            # callable = openai.callables_register.get_langchain_tool(name)
            # arguments = json.loads('{"name":"calculator","user_id":"abc"}')
            # result = callable.invoke(arguments)
            # return result
            assistant_id = assistant_repository.get_by_name(request.assistant_name)
            if not assistant_id:
                assistant_api = AssistantApiBuilder.create(name=request.assistant_name,
                                                           instructions="你是一个可以根据需要展示不同用户界面的的智能助理，不要"
                                                                        "假设将哪些值插入函数，如果用户的要求模棱两可，请要求说明。")
                assistant_api.load_callables(openai.callables_register.callable_names)
            else:
                assistant_api = AssistantApiBuilder.retrieve(assistant_id)
            result = assistant_api.ask(user_id=user_id,content=request.content, topic_name=request.topic)
            return result
        else:
            return await LLM().aask(request.content)
    else:
        return f"user_id {user_id} not found"
