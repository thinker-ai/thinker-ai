import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
role_router = APIRouter()
role_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'role')
role_dir = Jinja2Templates(directory=role_root)


@role_router.get("/role", response_class=HTMLResponse)
async def role(request: Request):
    return role_dir.TemplateResponse("role.html", {"request": request})