import os
import threading
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
deploy_directory = os.path.join(current_dir, 'gradio/')
# 确保部署目录存在
if not os.path.exists(deploy_directory):
    os.makedirs(deploy_directory)


def get_deploy_path(title, user_id: Optional[str] = None) -> str:
    if user_id is None:
        return os.path.join(deploy_directory, f"{title}.py")
    else:
        return os.path.join(deploy_directory, f"user_{user_id}/{title}.py")


def get_mount_path(title, user_id: Optional[str] = None) -> str:
    if user_id is None:
        return f"/tasks/{title}"
    else:
        return f"/tasks/user_{user_id}/{title}"


class Service:
    def __init__(self, name, user_id: Optional[str] = None):
        self.deploy_path = get_deploy_path(name, user_id)
        self.mount_path = get_mount_path(name, user_id)
        # 在asyncio.events.py中的get_event_loop方法中，要求
        # threading.current_thread() is threading.main_thread()
        # 才能执行self.set_event_loop(self.new_event_loop()，否则event_loop为空
        if threading.current_thread() is threading.main_thread():
            with open(self.deploy_path, 'r') as file:
                gradio_code = file.read()
            # 动态执行 Gradio 源代码
            if gradio_code:
                local_vars = {}
                exec(gradio_code, {}, local_vars)
                gr_blocks = local_vars.get(name)
                if gr_blocks:
                    self.gr_blocks = gr_blocks
                else:
                    raise RuntimeError("No interface defined")
        else:
            raise RuntimeError("current thread not main thread")
