from thinker_ai.agent.document_store.vector_database import VectorDatabase
from thinker_ai.agent.memory.humanoid_memory.memory_network import MemoryNetwork
from typing import List
from thinker_ai.agent.tools.embedding_helper import EmbeddingHelper


class DeepLearningMemoryNetwork(MemoryNetwork):
    def __init__(self, vector_db: VectorDatabase):
        """
        初始化记忆网络，并连接到向量数据库。
        :param vector_db: 用于存储和检索嵌入的向量数据库。
        """
        self.vector_db = vector_db
        self.embedding_helper = EmbeddingHelper()

    def add_memory(self, inputs: List[str]):
        """
        将新的信息添加到记忆网络中，向 `VectorDatabase` 中插入新的嵌入。
        :param inputs: 要添加的记忆文本列表。
        """
        self.vector_db.insert_batch(inputs)

    def query(self, question: str) -> str:
        """
        根据输入的问题，在记忆网络中检索最相关的记忆文本。
        :param question: 查询的问题。
        :return: 返回最相关的记忆文本。
        """
        results = self.vector_db.search(question, top_k=1)
        if results:
            return results[0][0]  # 返回最相关的文本
        return "No relevant information found."

    def is_related(self, text1: str, text2: str, similarity_threshold: float) -> bool:
        """
        判断两个文本在记忆网络中的相关性，通过余弦相似度判断。
        :param text1: 第一个文本。
        :param text2: 第二个文本。
        :param similarity_threshold: 相似度阀值
        :return: 如果两个文本相关，返回 True，否则返回 False。
        """
        return self.embedding_helper.is_related(text1, text2, similarity_threshold)

    def clear_memory(self):
        """
        清除记忆网络中的所有信息，清空 `VectorDatabase`。
        """
        self.vector_db.clear()
