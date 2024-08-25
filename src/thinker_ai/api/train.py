import os
from fastapi import APIRouter
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request
from thinker_ai.configs.const import PROJECT_ROOT
train_router = APIRouter()
train_root = os.path.join(PROJECT_ROOT, 'web', 'html', 'train')
train_dir = Jinja2Templates(directory=train_root)


@train_router.get("/train", response_class=HTMLResponse)
async def train(request: Request):
    return train_dir.TemplateResponse("train.html", {"request": request})