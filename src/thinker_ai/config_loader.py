import os
from pathlib import Path
import yaml
from thinker_ai.context import get_project_root
def load_config():
    with open(str(get_project_root() / "config.yaml"), "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
        if yaml_data:
            os.environ.update(yaml_data)
# 加载本地 YAML 文件
load_config()
