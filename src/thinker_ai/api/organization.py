import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
organization_router = APIRouter()
organization_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'organization')
organization_dir = Jinja2Templates(directory=organization_root)


@organization_router.get("/organization", response_class=HTMLResponse)
async def organization(request: Request):
    return organization_dir.TemplateResponse("organization.html", {"request": request})