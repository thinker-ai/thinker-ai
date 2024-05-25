import os
from fastapi import APIRouter, Depends
from fastapi import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.templating import Jinja2Templates
from thinker_ai.login import get_session
from thinker_ai.context import get_project_root
from thinker_ai.tasks.dynamic.service_loader import load_service
from thinker_ai.web_socket import enqueue_message

chat_router = APIRouter()

root_dir = get_project_root()
template_dir = os.path.join(root_dir, 'web', 'templates')
templates = Jinja2Templates(directory=template_dir)
# 添加调试信息
print(f"Template directory: {template_dir}")


class ChatRequest(BaseModel):
    topic: str
    content: str


@chat_router.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@chat_router.get("/mermaid", response_class=HTMLResponse)
async def mermaid(request: Request):
    return templates.TemplateResponse("mermaid.html", {"request": request})


def to_assistant_id(user_id, topic) -> str:
    return "asst_zBrqXNoQIvnX1TyyVry9UveZ"  #待替换


@chat_router.post("/chat", response_model=str)
async def chat(request: ChatRequest, session: dict = Depends(get_session)) -> str:
    user_id = session.get("user_id")
    mount_path = await load_service(user_id, "demo")
    message = {
        "port":"8000",
        "title":"demo",
        "mount_path":mount_path
    }
    enqueue_message(user_id, message)
    # agent = AssistantAgent.from_id(to_assistant_id(user_id, request.topic))
    # return agent.ask(request.content)
    return "ok"
