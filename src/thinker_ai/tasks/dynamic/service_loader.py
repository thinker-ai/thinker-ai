import asyncio

from fastapi import APIRouter

from thinker_ai.app_instance import app, is_route_mounted
from thinker_ai_tests.tasks.dynamic.demo_main import main_loop
from thinker_ai.tasks.dynamic.service_deployer import ServiceDeployer

loader_router = APIRouter()


@loader_router.get("/load_service", response_model=str)
async def load_service(user_id: str, title: str) -> str:
    if not is_route_mounted(app, "/load_service"):
        raise Exception("not this app")

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def on_main_thread():
        try:
            task_deployer = ServiceDeployer()
            result = task_deployer.load_service(app, user_id, title)
            loop.call_soon_threadsafe(future.set_result, result)
        except Exception as e:
            loop.call_soon_threadsafe(future.set_exception, e)

    main_loop.call_soon_threadsafe(on_main_thread)
    try:
        result = await future  # 等待任务完成
    except Exception as e:
        return str(e)
    return result
