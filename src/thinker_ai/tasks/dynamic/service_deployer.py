import os
from typing import Dict

import gradio as gr
from fastapi import FastAPI
from starlette.routing import Mount

from thinker_ai.tasks.dynamic.service import Service, get_deploy_path


def is_route_mounted(app: FastAPI, mount_path: str) -> bool:
    for route in app.routes:
        if isinstance(route, Mount) and route.path.startswith(mount_path):
            return True
    return False


class ServiceDeployer:
    unmounted_services_dict: Dict[str, Service] = {}

    def deploy_service(self, user_id, title, content) -> str:
        deploy_path = get_deploy_path(user_id, title)
        os.makedirs(os.path.dirname(deploy_path), exist_ok=True)
        with open(deploy_path, 'w') as file:
            file.write(content)
        return deploy_path

    def load_service(self, app: FastAPI, user_id, title)->str:
        service = Service(user_id, title)
        # 如果服务已加载，则卸载它
        if is_route_mounted(app, service.mount_path):
            self._unmount_gr_blocks(app, service.mount_path)
        self._mount_gr_blocks(app, service)
        return service.mount_path

    def _mount_gr_blocks(self, app: FastAPI, service: Service):
        app = gr.mount_gradio_app(app, service.gr_blocks, service.mount_path)
        # Blocks的处理队列只在系统启动的时候才会被执行一次startup_events，
        # 这导致启动后mount的Blocks的处理队列需要手动执行startup_events
        service.gr_blocks.startup_events()
        if not is_route_mounted(app, service.mount_path):
            raise Exception(f"Failed to mount service at {service.mount_path}")

    def _unmount_gr_blocks(self, app: FastAPI, mount_path):
        service = self.unmounted_services_dict.get(mount_path)
        if service:
            # 停止处理队列
            service.gr_blocks.close(True)
            # 移除已加载服务
            app.router.routes = [
                route for route in app.router.routes
                if not route.path.startswith(mount_path)
            ]
