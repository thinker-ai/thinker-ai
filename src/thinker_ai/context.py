from thinker_ai.configs.const import PROJECT_ROOT

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from thinker_ai.configs.config import Config
from thinker_ai.configs.llm_config import LLMConfig, LLMType
from thinker_ai.agent.provider.base_llm import BaseLLM
from thinker_ai.agent.provider.llm_provider_registry import create_llm_instance
from thinker_ai.agent.provider.cost_manager import (
    CostManager,
    FireworksCostManager,
    TokenCostManager,
)
from thinker_ai.utils.git_repository import GitRepository
from thinker_ai.utils.project_repo import ProjectRepo


class AttrDict(BaseModel):
    """A dict-like object that allows access to keys as attributes, compatible with Pydantic."""

    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        return self.__dict__.get(key, None)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def set(self, key, val: Any):
        self.__dict__[key] = val

    def get(self, key, default: Any = None):
        return self.__dict__.get(key, default)

    def remove(self, key):
        if key in self.__dict__:
            self.__delattr__(key)


class Context(BaseModel):
    """Env context for ThinkerAI"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kwargs: AttrDict = AttrDict()
    config: Config = Config.default()

    repo: Optional[ProjectRepo] = None
    git_repo: Optional[GitRepository] = None
    src_workspace: Optional[Path] = None
    cost_manager: CostManager = CostManager()

    _llm: Optional[BaseLLM] = None

    # def __init__(self, organization_id: str, solution_name: str, /, **data: Any):
    #     super().__init__(**data)
    #     self.organization_id = organization_id
    #     self.solution_name = solution_name
    #
    # def get_workspace_path(self) -> Path:
    #     return PROJECT_ROOT / 'workspace' / self.organization_id / self.solution_name

    # def recreate_workspace(self) -> Tuple[Path, Path]:
    #     workspace = self.get_workspace_path()
    #     try:
    #         shutil.rmtree(workspace)
    #     except FileNotFoundError:
    #         pass
    #     workspace.mkdir(parents=True, exist_ok=True)
    #     docs_path = workspace / 'docs'
    #     resources_path = workspace / 'resources'
    #     docs_path.mkdir(parents=True, exist_ok=True)
    #     resources_path.mkdir(parents=True, exist_ok=True)
    #     return docs_path, resources_path
    #
    # def load_file_from_workspace(self, file_name: str) -> str:
    #     if file_name.startswith('/'):
    #         file_name = file_name[1:]  # 否则会误判为根路径
    #     file_dir = self.get_workspace_path()
    #     return self.load_file(file_dir, file_name)
    #
    # def load_all_files_from_workspace(self, dir_name: str) -> dict:
    #     workspace_path = self.get_workspace_path()
    #     dir_path: Path = workspace_path / dir_name
    #     if not dir_path.exists() or not dir_path.is_dir():
    #         raise ValueError(f"{dir_path} is not a valid directory")
    #     file_data = {}
    #     for file in dir_path.iterdir():
    #         if file.is_file():
    #             with open(file, 'r', encoding='utf-8') as f:
    #                 file_data[file.name] = f.read()
    #     return file_data

    def load_file_from_project(self, file_name: str) -> str:
        if file_name.startswith('/'):
            file_name = file_name[1:]  # 否则会误判为根路径
        file_dir = PROJECT_ROOT
        return self.load_file(file_dir, file_name)

    def load_file(self, file_dir, file_name):
        file = Path(file_dir) / file_name
        with open(file, 'r') as file:
            content = file.read()
        return content


    def new_environ(self):
        """Return a new os.environ object"""
        env = os.environ.copy()
        # i = self.options
        # env.update({k: v for k, v in i.items() if isinstance(v, str)})
        return env

    def _select_costmanager(self, llm_config: LLMConfig) -> CostManager:
        """Return a CostManager instance"""
        if llm_config.api_type == LLMType.FIREWORKS:
            return FireworksCostManager()
        elif llm_config.api_type == LLMType.OPEN_LLM:
            return TokenCostManager()
        else:
            return self.cost_manager

    def llm(self) -> BaseLLM:
        """Return a LLM instance, fixme: support cache"""
        # if self._llm is None:
        self._llm = create_llm_instance(self.config.llm)
        if self._llm.cost_manager is None:
            self._llm.cost_manager = self._select_costmanager(self.config.llm)
        return self._llm

    def llm_with_cost_manager_from_llm_config(self, llm_config: LLMConfig) -> BaseLLM:
        """Return a LLM instance, fixme: support cache"""
        # if self._llm is None:
        llm = create_llm_instance(llm_config)
        if llm.cost_manager is None:
            llm.cost_manager = self._select_costmanager(llm_config)
        return llm

    def serialize(self) -> Dict[str, Any]:
        """Serialize the object's attributes into a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing serialized data.
        """
        return {
            "workdir": str(self.repo.workdir) if self.repo else "",
            "kwargs": {k: v for k, v in self.kwargs.__dict__.items()},
            "cost_manager": self.cost_manager.model_dump_json(),
        }

    def deserialize(self, serialized_data: Dict[str, Any]):
        """Deserialize the given serialized data and update the object's attributes accordingly.

        Args:
            serialized_data (Dict[str, Any]): A dictionary containing serialized data.
        """
        if not serialized_data:
            return
        workdir = serialized_data.get("workdir")
        if workdir:
            self.git_repo = GitRepository(local_path=workdir, auto_init=True)
            self.repo = ProjectRepo(self.git_repo)
            src_workspace = self.git_repo.workdir / self.git_repo.workdir.name
            if src_workspace.exists():
                self.src_workspace = src_workspace
        kwargs = serialized_data.get("kwargs")
        if kwargs:
            for k, v in kwargs.items():
                self.kwargs.set(k, v)
        cost_manager = serialized_data.get("cost_manager")
        if cost_manager:
            self.cost_manager.model_validate_json(cost_manager)
