from thinker_ai.app.design.solution.solution import Solution


class SolutionRepository:
    def get(self, name) -> Solution:
        pass

    def get_current(self, user_id) -> Solution:
        pass

    def save_current(self, user_id, solution:Solution):
        pass
