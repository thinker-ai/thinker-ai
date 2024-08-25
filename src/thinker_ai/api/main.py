import asyncio
import os

from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import uvicorn
from fastapi import APIRouter
from starlette.routing import Route, WebSocketRoute

from thinker_ai.api.criterion import criterion_router
from thinker_ai.api.fast_api_instance import app
from thinker_ai.api.login import login_router
from thinker_ai.api.marketing import marketing_router
from thinker_ai.api.resources import resources_router
from thinker_ai.api.strategy import strategy_router
from thinker_ai.api.train import train_router
from thinker_ai.api.web_socket_server import socket_router, process_message_queue
from thinker_ai.api.works import works_router
from thinker_ai.tasks.dynamic.service_deployer import deploy_ui, DeployArgs
from thinker_ai.tasks.dynamic.service_loader import ServiceLoader, LoadArgs

from thinker_ai.agent.openai_assistant_api import openai
from thinker_ai.configs.const import PROJECT_ROOT

main_loop = asyncio.get_event_loop()
background_tasks = []

# 假设 PROJECT_ROOT 是你的项目根目录
web_root = os.path.join(PROJECT_ROOT, 'web')
# 将整个 web 目录挂载为静态文件目录，这样可以使用相对路径访问静态文件
app.mount("/static", StaticFiles(directory=web_root + "/static"), name="static")
app.mount("/script", StaticFiles(directory=web_root + "/script"), name="script")
service_loader = ServiceLoader(app=app, main_loop=main_loop)


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    main_dir = Jinja2Templates(directory=web_root)
    return main_dir.TemplateResponse("/html/main.html", {"request": request})


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    main_dir = Jinja2Templates(directory=web_root)
    return main_dir.TemplateResponse("/html/home.html", {"request": request})


# 1、不能在该文件之外执行 include_router 操作，因为当前文件不感知其它文件，导致 include_router 不会执行，
# 2、include_router 操作必须在所有 router 的地址映射完成之后执行
# 3、无法采用官方推荐的@asynccontextmanager注解方式，因为app的创建必须和@asynccontextmanager在一起，
# 而app实例放在该文件中被其它文件import的时候，会成为另一个不同的实例，导致@asynccontextmanager不得不和app的创建分离。
@app.on_event("startup")
async def startup():
    from chat import chat_router
    from design import design_router
    # 启动消息队列处理任务，并存储任务引用
    task = asyncio.create_task(process_message_queue())
    background_tasks.append(task)
    register_callables()
    include_router(chat_router)
    include_router(criterion_router)
    include_router(marketing_router)
    include_router(resources_router)
    include_router(strategy_router)
    include_router(train_router)
    include_router(works_router)
    include_router(design_router)
    include_router(login_router)
    include_router(socket_router)


def include_router(router: APIRouter):
    if not is_router_included(router):
        app.include_router(router)


def register_callables():
    openai.callables_register.register_callable(service_loader.load_ui_and_show, LoadArgs)
    openai.callables_register.register_callable(deploy_ui, DeployArgs)


def is_router_included(router: APIRouter) -> bool:
    # 获取 router 中所有路由的路径
    router_routes = {route.path for route in router.routes if isinstance(route, (Route, WebSocketRoute))}
    # 获取 app 中所有路由的路径
    app_routes = {route.path for route in app.routes if isinstance(route, (Route, WebSocketRoute))}
    # 判断 router 的路径集合与 app 的路径集合是否有交集
    return not router_routes.isdisjoint(app_routes)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
