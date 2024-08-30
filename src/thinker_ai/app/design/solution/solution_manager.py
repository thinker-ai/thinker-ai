import uuid
from typing import Optional

from thinker_ai.app.design.solution.solution import Solution
from thinker_ai.app.design.solution.solution_repository import SolutionRepository
from thinker_ai.configs.config import config


class SolutionManager:
    solution_repository = SolutionRepository(base_dir=str(config.workspace.path / "data/"), file_name=config.workspace.user_solutions_file)

    def get_not_done(self, user_id: str) -> Optional[Solution]:
        if user_id:
            current = self.solution_repository.get_not_done(user_id)
            if current is None:
                return Solution(id=str(uuid.uuid4()), user_id=user_id)
        return None

    def save(self, solution: Solution):
        if solution.id and solution.user_id:
            self.solution_repository.set(solution)
            self.solution_repository.save()
