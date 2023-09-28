import shutil
from pathlib import Path
from typing import Tuple

from thinker_ai.actor import Actor


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


class Context:

    def __init__(self, organization_id: str, actor: Actor, solution_name: str):
        self.organization_id = organization_id
        self.actor = actor
        self.solution_name = solution_name

    def get_workspace_path(self) -> Path:
        return get_project_root() / 'workspace' / self.organization_id / self.solution_name

    def recreate_workspace(self) -> Tuple[Path, Path]:
        workspace = self.get_workspace_path()
        try:
            shutil.rmtree(workspace)
        except FileNotFoundError:
            pass
        workspace.mkdir(parents=True, exist_ok=True)
        docs_path = workspace / 'docs'
        resources_path = workspace / 'resources'
        docs_path.mkdir(parents=True, exist_ok=True)
        resources_path.mkdir(parents=True, exist_ok=True)
        return docs_path, resources_path

    def load_file_from_workspace(self, file_name: str) -> str:
        if file_name.startswith('/'):
            file_name=file_name[1:]#否则会误判为根路径
        file_dir = self.get_workspace_path()
        return self.load_file(file_dir, file_name)

    def load_all_files_from_workspace(self, dir_name: str)-> dict:
        workspace_path = self.get_workspace_path()
        dir_path:Path = workspace_path / dir_name
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a valid directory")
        file_data = {}
        for file in dir_path.iterdir():
            if file.is_file():
                with open(file, 'r', encoding='utf-8') as f:
                    file_data[file.name] = f.read()
        return file_data

    def load_file_from_project(self, file_name: str) -> str:
        if file_name.startswith('/'):
            file_name=file_name[1:]#否则会误判为根路径
        file_dir = get_project_root()
        return self.load_file(file_dir, file_name)

    def load_file(self, file_dir, file_name):
        file = Path(file_dir) / file_name
        with open(file, 'r') as file:
            content = file.read()
        return content
