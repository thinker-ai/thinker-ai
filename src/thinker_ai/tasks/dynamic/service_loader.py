import asyncio
import threading
from typing import Optional, Dict, Type
import gradio as gr
from fastapi import APIRouter, FastAPI
from starlette.routing import Route, Mount

from thinker_ai.app_instance import app
from thinker_ai.tasks.dynamic.service import Service
from thinker_ai.web_socket import send_message_to_client
from thinker_ai_tests.tasks.dynamic.demo_main import main_loop

loader_router = APIRouter()


def is_route_mounted(app: FastAPI, mount_path: str) -> bool:
    for route in app.routes:
        if isinstance(route, (Mount, APIRouter, Route)) and route.path.startswith(mount_path):
            return True
    return False


@loader_router.get("/load_service", response_model=str)
def load_service_and_push_to_user(name: str, user_id: Optional[str] = None) -> str:
    """
    将gradio代码加载运行后把地址推给用户
    :param name: 加载文件名
    :param user_id: 用户id
    :return: 加载的服务的挂载地址
    """
    mount_path = load_service_on_main_thread(name, user_id)
    message = {
        "port": 8000,
        "name": name,
        "mount_path": mount_path
    }
    send_message_to_client(user_id, message)
    return mount_path


def load_service_on_main_thread(name: str, user_id: Optional[str] = None) -> str:
    if not is_route_mounted(app, "/load_service"):
        raise Exception("not this app")

    result_event = threading.Event()
    result_container = {}

    def on_main_thread():
        try:
            service_loader = ServiceLoader()
            mount_path = service_loader.load_service_dynamic(app, name, user_id)
            result_container["result"] = mount_path
        except Exception as e:
            result_container["exception"] = e
        finally:
            result_event.set()

    if threading.current_thread() is threading.main_thread():
        on_main_thread()
    else:
        main_loop.call_soon_threadsafe(on_main_thread)

    result_event.wait()  # 等待任务完成

    if result_container.get("exception") is not None:
        return str(result_container.get("exception"))

    return result_container.get("result")


class ServiceLoader:
    mounted_services_dict: Dict[str, Service] = {}

    def load_service_static(self, app: FastAPI, name, user_id: Optional[str] = None) -> str:
        return self._load_service(app, name, user_id, True)

    def load_service_dynamic(self, app: FastAPI, name, user_id: Optional[str] = None) -> str:
        return self._load_service(app, name, user_id, False)

    def _load_service(self, app: FastAPI, name, user_id: Optional[str] = None, static_mount: bool = False) -> str:
        if threading.current_thread() is threading.main_thread():
            service = Service(name, user_id)
            # 如果服务已加载，则卸载它
            if is_route_mounted(app, service.mount_path):
                self._unmount_gr_blocks(app, service.mount_path)
            self._mount_gr_blocks(app, service, static_mount)
            return service.mount_path
        else:
            raise Exception("not in main_thread")

    def _mount_gr_blocks(self, app: FastAPI, service: Service, static_mount: bool) -> None:
        app = gr.mount_gradio_app(app, service.gr_blocks, service.mount_path)
        if not static_mount:
            # Blocks的处理队列只在编译期静态mount的时候才会被执行startup_events，
            # 这导致非静态的mount的Blocks的处理队列需要手动执行startup_events
            service.gr_blocks.startup_events()
        self.mounted_services_dict[service.mount_path] = service
        if not is_route_mounted(app, service.mount_path):
            raise Exception(f"Failed to mount service at {service.mount_path}")

    def _unmount_gr_blocks(self, app: FastAPI, mount_path):
        service = self.mounted_services_dict.pop(mount_path)

        if service:
            # 停止处理队列
            service.gr_blocks.close(True)
            # 移除已加载服务
            found = None
            for route in app.routes:
                if isinstance(route, Mount) and route.path.startswith(mount_path):
                    found = route
            if found:
                app.routes.remove(found)
