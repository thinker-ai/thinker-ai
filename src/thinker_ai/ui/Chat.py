import os
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from thinker_ai.context import get_project_root

chat_router = APIRouter()

root_dir = get_project_root()
template_dir = os.path.join(root_dir, 'web','templates')
templates = Jinja2Templates(directory=template_dir)


@chat_router.get("/", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@chat_router.get("/mermaid", response_class=HTMLResponse)
async def mermaid(request: Request):
    return templates.TemplateResponse("mermaid.html", {"request": request})
# @chat_router.post("/chat_to", response_model=DataModel)
# async def chat_to(human_data: DataModel) -> DataModel:
#     [content] = human_data
#     speaker = SpeakAgent("speaker")
#     return await speaker.ask(content)
