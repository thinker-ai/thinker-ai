import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
criterion_router = APIRouter()
criterion_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'criterion')
criterion_dir = Jinja2Templates(directory=criterion_root)


@criterion_router.get("/criterion", response_class=HTMLResponse)
async def criterion(request: Request):
    return criterion_dir.TemplateResponse("criterion.html", {"request": request})
