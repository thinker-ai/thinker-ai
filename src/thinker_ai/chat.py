import os
from fastapi import APIRouter, Depends
from fastapi import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.templating import Jinja2Templates

from thinker_ai.agent.assistant_api_factory import AssistantApiFactory, PROVIDER
from thinker_ai.agent.assistant_api_repository import AssistantRepository
from thinker_ai.agent.openai_assistant_api.openai_assistant_api_repository import OpenAiAssistantRepository
from thinker_ai.agent.provider import llm
from thinker_ai.login import get_session
from thinker_ai.configs.const import PROJECT_ROOT

chat_router = APIRouter()
root_dir = PROJECT_ROOT
template_dir = os.path.join(root_dir, 'web', 'templates')
templates = Jinja2Templates(directory=template_dir)
assistant_repository: AssistantRepository = OpenAiAssistantRepository.get_instance()

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
            assistant_api = assistant_repository.get_assistant_api_by_name(user_id=user_id, name=request.assistant_name)
            if not assistant_api:
                assistant_api = AssistantApiFactory.create(provider=PROVIDER.OpenAILLM,
                                                           user_id=user_id,
                                                           name=request.assistant_name,
                                                           instructions="你是一个可以回答任何问题的智能助理")
                assistant_repository.add_assistant_api(assistant_api=assistant_api, user_id=user_id)
            return assistant_api.ask(content=request.content, topic=request.topic)
        else:
            return await llm.LLM().aask(request.content)
    else:
        return f"user_id {user_id} not found"
