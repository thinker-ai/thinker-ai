from thinker_ai.context import Context


class FileLoader:
    def __init__(self,context:Context):
        self.context=context

    def load_file(self, file_dir: str) -> str:
        file_path = self.context.get_workspace_path()
        file = file_path / file_dir
        with open(file, 'r') as file:
            content = file.read()
        return content
