import os
import threading
from importlib import util
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


def load_module_from_file(filepath, module_name):
    spec = util.spec_from_file_location(module_name, filepath)
    module = util.module_from_spec(spec)
    with open(filepath, 'r') as file:
        code = file.read()
    exec(code, module.__dict__)
    return module


class Service:
    def __init__(self, name, user_id: Optional[str] = None):
        self.deploy_path = get_deploy_path(name, user_id)
        self.mount_path = get_mount_path(name, user_id)

        if threading.current_thread() is threading.main_thread():
            # 动态加载模块
            self.module = load_module_from_file(self.deploy_path, name)
            # 获取 Gradio Blocks 实例
            self.gr_blocks = getattr(self.module, name, None)
            if not self.gr_blocks:
                raise RuntimeError("未定义接口")
        else:
            raise RuntimeError("当前线程不是主线程")
