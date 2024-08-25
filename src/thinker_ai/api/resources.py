import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
resources_router = APIRouter()
resources_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'resources')
resources_dir = Jinja2Templates(directory=resources_root)


@resources_router.get("/resources", response_class=HTMLResponse)
async def resources(request: Request):
    return resources_dir.TemplateResponse("resources.html", {"request": request})