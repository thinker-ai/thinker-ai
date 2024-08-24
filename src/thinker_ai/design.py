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
async def main(request: Request):
    return design_dir.TemplateResponse("main.html", {"request": request})


@design_router.get("/design/list", response_class=HTMLResponse)
async def design_list(request: Request):
    return design_dir.TemplateResponse("list.html", {"request": request})


@design_router.get("/design/one", response_class=HTMLResponse)
async def design_one(request: Request):
    return design_dir.TemplateResponse("one.html", {"request": request})


@design_router.get("/design/one/requirements", response_class=HTMLResponse)
async def design_one_requirements(request: Request):
    return design_dir.TemplateResponse("one/requirements.html", {"request": request})


@design_router.get("/design/one/knowledge", response_class=HTMLResponse)
async def design_one_knowledge(request: Request):
    return design_dir.TemplateResponse("one/knowledge.html", {"request": request})


@design_router.get("/design/one/resource", response_class=HTMLResponse)
async def design_one_resource(request: Request):
    return design_dir.TemplateResponse("one/resource.html", {"request": request})


@design_router.get("/design/one/criterion", response_class=HTMLResponse)
async def design_one_criterion(request: Request):
    return design_dir.TemplateResponse("one/criterion.html", {"request": request})


@design_router.get("/design/one/solution", response_class=HTMLResponse)
async def design_one_solution(request: Request):
    return design_dir.TemplateResponse("one/solution.html", {"request": request})


@design_router.get("/design/one/strategy", response_class=HTMLResponse)
async def design_one_strategy(request: Request):
    return design_dir.TemplateResponse("one/strategy.html", {"request": request})
