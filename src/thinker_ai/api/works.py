import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
works_router = APIRouter()
works_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'works')
works_dir = Jinja2Templates(directory=works_root)


@works_router.get("/works", response_class=HTMLResponse)
async def works(request: Request):
    return works_dir.TemplateResponse("works.html", {"request": request})