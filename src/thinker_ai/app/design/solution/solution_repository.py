from thinker_ai.app.design.solution.solution import Solution


class SolutionRepository:
    def get(self, solution_id) -> Solution:
        pass

    def get_current(self, user_id) -> Solution:
        pass
