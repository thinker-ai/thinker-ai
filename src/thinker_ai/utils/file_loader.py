from thinker_ai.context import Context


class FileLoader:
    def __init__(self,context:Context):
        self.context=context

    def load_file(self, file_dir: str) -> str:
        solution_path = self.context.get_workspace_path()
        prd_file = solution_path / file_dir
        with open(prd_file, 'r') as file:
            prd = file.read()
        return prd
