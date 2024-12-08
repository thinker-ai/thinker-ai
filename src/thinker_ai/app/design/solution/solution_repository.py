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

    def get_by_user(self, user_id) -> list:
        result = []
        for solution_dict in self.solutions_dict.values():
            solution = Solution.from_dict(solution_dict)
            if solution_dict.get("user_id") == user_id:
                result.append({
                    "id": solution.id,
                    "name": solution.name,
                    "done": solution.done,
                })
        return result

    def set(self, solution: Solution):
        self.solutions_dict[solution.id] = solution.to_dict()

    def to_file(self, base_dir: str, file_name: str):
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as file:
            try:
                json.dump(self.solutions_dict, file, indent=2, ensure_ascii=False)
            except JSONDecodeError as e:
                print(f"Error while writing to file: {e}")

    def save(self):
        if self.base_dir and self.file_name:
            self.to_file(self.base_dir, self.file_name)
        else:
            raise Exception('No base dir or file_name specified')
