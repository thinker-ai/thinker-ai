import asyncio
import os

import uvicorn
from fastapi import APIRouter
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from thinker_ai.configs.const import PROJECT_ROOT
from thinker_ai.app_instance import app
from thinker_ai.tasks.dynamic.service_deployer import deploy_ui, DeployArgs
from thinker_ai.tasks.dynamic.service_loader import ServiceLoader, LoadArgs
from thinker_ai.web_socket_server import process_message_queue
from thinker_ai.agent.openai_assistant_api import openai
# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, 'web', 'static')), name="static")
app.mount("/script", StaticFiles(directory=os.path.join(PROJECT_ROOT, 'web', 'script')), name="script")
app.mount("/css", StaticFiles(directory=os.path.join(PROJECT_ROOT, 'web', 'css')), name="css")
main_loop = asyncio.get_event_loop()
background_tasks = []


# 1、不能在该文件之外执行 include_router 操作，因为当前文件不感知其它文件，导致 include_router 不会执行，
# 2、include_router 操作必须在所有 router 的地址映射完成之后执行
# 3、无法采用官方推荐的@asynccontextmanager注解方式，因为app的创建必须和@asynccontextmanager在一起，
# 而app实例放在该文件中被其它文件import的时候，会成为另一个不同的实例，导致@asynccontextmanager不得不和app的创建分离。
@app.on_event("startup")
async def startup():
    from chat import chat_router
    from thinker_ai.login import login_router
    from thinker_ai.web_socket_server import socket_router
    # 启动消息队列处理任务，并存储任务引用
    task = asyncio.create_task(process_message_queue())
    background_tasks.append(task)
    register_callables()
    if not is_router_included(chat_router):
        app.include_router(chat_router)
    if not is_router_included(login_router):
        app.include_router(login_router)
    if not is_router_included(socket_router):
        app.include_router(socket_router)


def register_callables():
    service_loader = ServiceLoader(app=app, main_loop=main_loop)
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
