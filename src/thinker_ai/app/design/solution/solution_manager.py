import uuid
from typing import Optional

from thinker_ai.app.design.solution.solution import Solution
from thinker_ai.app.design.solution.solution_repository import SolutionRepository
from thinker_ai.configs.config import config


class SolutionManager:
    solution_repository = SolutionRepository(base_dir=str(config.workspace.path / "data/"), file_name=config.workspace.user_solutions_file)

    def get_by_id(self, user_id:str, id:str) -> Optional[Solution]:
        current = self.solution_repository.get(id)
        if current is None:
            current = Solution(id=str(uuid.uuid4()), user_id=user_id)
        return current

    def save(self, solution: Solution):
        if solution.id and solution.user_id:
            self.solution_repository.set(solution)
            self.solution_repository.save()

    def get_solutions_list(self, user_id)->list:
        solutions_list = self.solution_repository.get_by_user(user_id)
        return solutions_list
