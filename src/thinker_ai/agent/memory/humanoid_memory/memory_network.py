from typing import List


class MemoryNetwork:
    def __init__(self):
        # 初始化记忆网络的参数和模型
        pass

    def add_memory(self, inputs: List[str]):
        """
        将新的信息添加到记忆网络中。
        """
        pass

    def query(self, question: str) -> str:
        """
        根据输入的问题，在记忆网络中检索相关的答案或信息。
        """
        pass

    def is_related(self, text1: str, text2: str) -> bool:
        """
        判断两个文本在记忆网络中是否相关。
        """
        pass

    def clear_memory(self):
        """
        清除记忆网络中的所有信息。
        """
        pass