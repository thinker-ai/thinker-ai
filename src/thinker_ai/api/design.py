import os
from fastapi import APIRouter, Depends
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.api.login import get_session
from thinker_ai.configs.const import PROJECT_ROOT

design_router = APIRouter()
design_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'design')
design_dir = Jinja2Templates(directory=design_root)


@design_router.get("/design", response_class=HTMLResponse)
async def main(request: Request):
    return design_dir.TemplateResponse("design.html", {"request": request})


@design_router.get("/design/list", response_class=HTMLResponse)
async def design_list(request: Request):
    return design_dir.TemplateResponse("list.html", {"request": request})


@design_router.get("/design/one", response_class=HTMLResponse)
async def design_one(request: Request):
    return design_dir.TemplateResponse("one.html", {"request": request})


@design_router.get("/design/one/criterion", response_class=HTMLResponse)
async def design_one_criterion(request: Request):
    return design_dir.TemplateResponse("one/criterion.html", {"request": request})


@design_router.get("/design/one/solution", response_class=HTMLResponse)
async def design_one_solution(request: Request):
    return design_dir.TemplateResponse("one/solution.html", {"request": request})


@design_router.get("/design/one/strategy", response_class=HTMLResponse)
async def design_one_strategy(request: Request):
    return design_dir.TemplateResponse("one/strategy.html", {"request": request})


@design_router.get("/design/one/resources", response_class=HTMLResponse)
async def design_one_resource(request: Request):
    return design_dir.TemplateResponse("one/resources.html", {"request": request})


@design_router.get("/design/one/resources/tools", response_class=HTMLResponse)
async def design_one_resources_tools(request: Request):
    return design_dir.TemplateResponse("one/resources/tools.html", {"request": request})


@design_router.get("/design/one/resources/solutions", response_class=HTMLResponse)
async def design_one_resources_solutions(request: Request):
    return design_dir.TemplateResponse("one/resources/solutions.html", {"request": request})


@design_router.get("/design/one/resources/trains", response_class=HTMLResponse)
async def design_one_resources_trains(request: Request):
    return design_dir.TemplateResponse("one/resources/trains.html", {"request": request})


@design_router.get("/design/one/resources/data_sources", response_class=HTMLResponse)
async def design_one_resources_data_sources(request: Request):
    return design_dir.TemplateResponse("one/resources/data_sources.html", {"request": request})


@design_router.get("/design/one/resources/third_parties", response_class=HTMLResponse)
async def design_one_resources_third_party(request: Request):
    return design_dir.TemplateResponse("one/resources/third_parties.html", {"request": request})


@design_router.get("/design/one/solution/current", response_model=str)
async def design_one_solution_current(request: Request, session: dict = Depends(get_session)) -> str:
    user_id = session.get("user_id")

    return user_id
