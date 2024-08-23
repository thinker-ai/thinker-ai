import os

from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request

from thinker_ai.configs.const import PROJECT_ROOT

design_router = APIRouter()

design_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'design')
design_dir = Jinja2Templates(directory=design_root)


@design_router.get("/design", response_class=HTMLResponse)
async def design(request: Request):
    return design_dir.TemplateResponse("main.html", {"request": request})


@design_router.get("/design/list", response_class=HTMLResponse)
async def design(request: Request):
    return design_dir.TemplateResponse("list.html", {"request": request})


@design_router.get("/design/create", response_class=HTMLResponse)
async def design(request: Request):
    return design_dir.TemplateResponse("create.html", {"request": request})
