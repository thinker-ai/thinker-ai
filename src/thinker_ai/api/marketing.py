import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
marketing_router = APIRouter()
marketing_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'marketing')
marketing_dir = Jinja2Templates(directory=marketing_root)


@marketing_router.get("/marketing", response_class=HTMLResponse)
async def marketing(request: Request):
    return marketing_dir.TemplateResponse("marketing.html", {"request": request})