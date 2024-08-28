from thinker_ai.app.design.solution.solution import Solution
from thinker_ai.app.design.solution.solution_repository import SolutionRepository


class SolutionManager:
    solution_repository = SolutionRepository()

    def get_current_solution(self, user_id: str) -> Solution:
        current = self.solution_repository.get_current(user_id)
        if current is None:
            return Solution(user_id=user_id)
