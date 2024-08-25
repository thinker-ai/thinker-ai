import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
strategy_router = APIRouter()
strategy_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'strategy')
strategy_dir = Jinja2Templates(directory=strategy_root)


@strategy_router.get("/strategy", response_class=HTMLResponse)
async def strategy(request: Request):
    return strategy_dir.TemplateResponse("strategy.html", {"request": request})