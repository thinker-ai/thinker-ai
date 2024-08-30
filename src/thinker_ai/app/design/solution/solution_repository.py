import json
import os
from json import JSONDecodeError
from typing import Optional
from thinker_ai.app.design.solution.solution import Solution


class SolutionRepository:
    def __init__(self, base_dir: str, file_name: str):
        self.base_dir = base_dir
        self.file_name = file_name
        self.solutions_dict = {}
        file_path = os.path.join(base_dir, file_name)
        os.makedirs(base_dir, exist_ok=True)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    self.solutions_dict = json.load(file)
            except JSONDecodeError:
                pass

    def get(self, id) -> Optional[Solution]:
        solution_dict = self.solutions_dict.get(id)
        if solution_dict:
            return Solution.from_dict(solution_dict)
        return None

    def set(self, solution: Solution):
        self.solutions_dict[solution.id] = solution.to_dict()

    def get_not_done(self, user_id) -> Optional[Solution]:
        for solution_dict in self.solutions_dict.values():
            if solution_dict.get("user_id") == user_id and not solution_dict.get("done"):
                return Solution.from_dict(solution_dict)
        return None

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.solutions_dict, file, indent=2, ensure_ascii=False)

    def save(self):
        if self.base_dir and self.file_name:
            self.to_file(self.base_dir, self.file_name)
        else:
            raise Exception('No base dir or file_name specified')
