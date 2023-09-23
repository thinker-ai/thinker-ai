import os
from pathlib import Path
import yaml

def get_project_root() -> Path:
    """逐级向上寻找项目根目录"""
    current_path = Path.cwd()
    while True:
        if (current_path / '.git').exists() or \
                (current_path / '.project_root').exists() or \
                (current_path / '.gitignore').exists():
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            raise Exception("Project root not found.")
        current_path = parent_path


configs: dict = {}
configs.update(os.environ)
# 加载本地 YAML 文件
with open(str(get_project_root() / "config.yaml"), "r", encoding="utf-8") as file:
    yaml_data = yaml.safe_load(file)
    if yaml_data:
        configs.update(yaml_data)
