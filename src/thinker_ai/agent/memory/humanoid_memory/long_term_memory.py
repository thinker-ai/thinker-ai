from typing import Dict, Optional, List

from thinker_ai.agent.memory.humanoid_memory.differentiable_neural_computer import DifferentiableNeuralComputer


class LongTermMemory:
    """
    长期记忆模块，用于存储和检索长期知识和信息。
    """

    def __init__(self, dnc:DifferentiableNeuralComputer):
        self.knowledge_base: Dict[str, str] = {}
        self.dnc = dnc  # 初始化 DNC

    def save(self):
        """
        保存长期记忆。
        """
        self.persistence.save(self.knowledge_base)

    def load(self):
        """
        加载长期记忆。
        """
        data = self.persistence.load()
        if data:
            self.knowledge_base = data
            # 将知识加载到 DNC
            for value in self.knowledge_base.values():
                self.dnc.store([value])
        else:
            self.knowledge_base = {}

    def store_message(self, key: str, value: str):
        self.knowledge_base[key] = value
        # 将知识存储到 DNC
        self.dnc.store([value])

    def retrieve_knowledge(self, key: str) -> Optional[str]:
        return self.knowledge_base.get(key)

    def search_knowledge(self, query: str) -> List[str]:
        # 从 DNC 中搜索相关知识
        result = self.dnc.search(query)
        return result if result else []

    def is_related(self, text1: str, text2: str) -> bool:
        # 使用 DNC 判断相关性
        return self.dnc.is_related(text1, text2)

    def rewrite(self, sentence: str, context: str) -> str:
        # 使用 DNC 进行重写（简化处理）
        return f"{context} {sentence}"

    def clear(self):
        self.knowledge_base.clear()
        self.dnc.clear_memory()