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
