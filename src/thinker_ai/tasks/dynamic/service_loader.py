import threading
from asyncio import AbstractEventLoop
from typing import Optional, Dict
import gradio as gr
from fastapi import APIRouter, FastAPI
from starlette.routing import Route, Mount
from langchain.pydantic_v1 import BaseModel, Field

from thinker_ai.api.web_socket_server import send_to_front
from thinker_ai.tasks.dynamic.service import Service


class LoadArgs(BaseModel):
    name: str = Field(
        ...,
        description="应用名称，即是gradio文件名"
    )
    user_id: str = Field(
        ...,
        description="用户id"
    )


class ServiceLoader:
    mounted_services_dict: Dict[str, Service] = {}

    def __init__(self, app: FastAPI, main_loop: AbstractEventLoop):
        self.app = app
        self.main_loop = main_loop

    def is_route_mounted(self, mount_path: str) -> bool:
        for route in self.app.routes:
            if isinstance(route, (Mount, APIRouter, Route)) and route.path.startswith(mount_path):
                return True
        return False

    def load_ui_and_show(self,name: str, user_id: Optional[str] = None) -> str:
        """
        将gradio代码加载运行后把地址推给用户
        :param name: 加载文件名
        :param user_id: 用户id
        :return: 加载的服务的挂载地址
        """
        mount_path = self.load_ui_on_main_thread(name, user_id)
        message = {
            "port": 8000,
            "name": name,
            "mount_path": mount_path
        }
        send_to_front(user_id, message)
        return mount_path

    def load_ui_on_main_thread(self, name: str, user_id: Optional[str] = None) -> str:
        result_event = threading.Event()
        result_container = {}

        def on_main_thread():
            try:
                mount_path = self.load_ui_dynamic(name, user_id)
                result_container["result"] = mount_path
            except Exception as e:
                result_container["exception"] = e
            finally:
                result_event.set()

        if threading.current_thread() is threading.main_thread():
            on_main_thread()
        else:
            self.main_loop.call_soon_threadsafe(on_main_thread)

        result_event.wait()  # 等待任务完成

        if result_container.get("exception") is not None:
            return str(result_container.get("exception"))

        return result_container.get("result")

    def load_ui_static(self, name, user_id: Optional[str] = None) -> str:
        return self._load_ui(name, user_id, True)

    def load_ui_dynamic(self, name, user_id: Optional[str] = None) -> str:
        return self._load_ui(name, user_id, False)

    def _load_ui(self, name, user_id: Optional[str] = None, static_mount: bool = False) -> str:
        if threading.current_thread() is threading.main_thread():
            service = Service(name, user_id)
            # 如果服务已加载，则卸载它
            if self.is_route_mounted(service.mount_path):
                self._unmount_gr_blocks(service.mount_path)
            self._mount_gr_blocks(service, static_mount)
            return service.mount_path
        else:
            raise Exception("not in main_thread")

    def _mount_gr_blocks(self, service: Service, static_mount: bool) -> None:
        gr.mount_gradio_app(self.app, service.gr_blocks, service.mount_path)
        if not static_mount:
            # Blocks的处理队列只在编译期静态mount的时候才会被执行startup_events，
            # 这导致非静态的mount的Blocks的处理队列需要手动执行startup_events
            service.gr_blocks.startup_events()
        self.mounted_services_dict[service.mount_path] = service
        if not self.is_route_mounted(service.mount_path):
            raise Exception(f"Failed to mount service at {service.mount_path}")

    def _unmount_gr_blocks(self, mount_path):
        service = self.mounted_services_dict.get(mount_path)
        if service:
            # 停止处理队列
            service.gr_blocks.close(True)
            # 移除已加载服务
        found = None
        for route in self.app.routes:
            if isinstance(route, Mount) and route.path.startswith(mount_path):
                found = route
        if found:
            self.app.routes.remove(found)
