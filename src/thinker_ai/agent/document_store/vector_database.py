import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from thinker_ai.agent.tools.embedding_helper import EmbeddingHelper


class VectorDatabase(ABC):

    @abstractmethod
    def insert(self, text: str, model: str = "text-embedding-3-small") -> None:
        pass

    @abstractmethod
    def insert_batch(self, texts: List[str], model: str = "text-embedding-3-small"):
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """检索与查询向量最相似的 top_k 向量"""
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> None:
        pass

    @abstractmethod
    def update(self, vector_id: str, text: str, model: str = "text-embedding-3-small") -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        """清除数据库中的所有向量"""
        pass


import faiss
from typing import List, Tuple, Optional


class FAISSVectorDatabase(VectorDatabase):
    # 在 __init__ 中增加一个 vectors 字典来存储向量数据
    # 增加一个 vector_id_list 来维护向量的顺序
    def __init__(self, dimension: int = 1536, rebuild_threshold: float = 0.3):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = {}  # 用于存储文本与向量 ID 的映射
        self.vectors = {}  # 用于存储向量数据
        self.vector_id_list = []  # 用于存储 vector_id 的顺序
        self.invalid_count = 0  # 用于记录无效向量的数量
        self.rebuild_threshold = rebuild_threshold  # 设定的阈值比例
        self.embedding_helper = EmbeddingHelper()

    def _search_with_embedding(self, vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:  # 检查索引中是否有向量
            return []

        query_vector = np.array(vector).astype('float32').reshape(1, -1)  # 确保向量的形状正确
        n = query_vector.shape[0]  # 查询向量的数量（通常是1）

        # 调用 FAISS 的 search 方法
        distances, labels = self.index.search(query_vector, top_k) #特殊的np类型，IDE会误报错

        results = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx != -1:
                vector_id = self.vector_id_list[idx]  # 使用 vector_id_list 来获取正确的向量 ID
                # 只返回标记为有效的向量
                if self.metadata[vector_id].get('is_valid', True):  # 默认向量有效
                    results.append((vector_id, dist))
        return results

    def insert(self, text: str, model: str = "text-embedding-3-small") -> None:
        embedding = self.embedding_helper.get_embedding_with_cache(text, model=model)
        vector_id = str(uuid.uuid4())  # 生成唯一ID
        self._insert_vector(vector_id, embedding, metadata={"text": text})

    # 在 _insert_vector 方法中插入 vector_id_list
    def _insert_vector(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        vector_np = np.array(vector).astype('float32').reshape(1, -1)  # 保证是 (1, d) 形式的 numpy 数组
        n = vector_np.shape[0]  # 向量的数量，这里是 1
        self.index.add(vector_np)  # IDE 会提示错误，忽略

        # 保存向量信息
        self.vectors[vector_id] = vector_np  # 保存向量到 vectors 字典
        self.vector_id_list.append(vector_id)  # 将 vector_id 添加到列表中，保持顺序
        if metadata:
            self.metadata[vector_id] = metadata
            self.metadata[vector_id]['is_valid'] = True

    def insert_batch(self, texts: List[str], model: str = "text-embedding-3-small"):
        embeddings = self.embedding_helper.get_embeddings(texts, model=model)
        vectors_np = np.array(embeddings).astype('float32')  # 将嵌入转为 numpy 数组

        # 计算 n: 向量的数量
        n = vectors_np.shape[0]  # n 是行数，表示有多少个向量

        self.index.add(vectors_np)  # IDE 会提示错误，忽略

        for text, embedding in zip(texts, embeddings):
            vector_id = str(uuid.uuid4())  # 使用 UUID 生成唯一 ID
            self.vectors[vector_id] = np.array(embedding)
            self.vector_id_list.append(vector_id)
            self.metadata[vector_id] = {"text": text, "is_valid": True}

    def search(self, query: str, top_k: int = 1) -> List[Tuple[str, str]]:
        """
        根据查询文本生成嵌入，并在数据库中搜索相关文本。
        返回格式为 [(文本ID, 相关文本)]。
        """
        query_embedding = self.embedding_helper.get_embedding_with_cache(query)
        results = self._search_with_embedding(query_embedding, top_k=top_k)
        return [(self.metadata.get(vector_id, {}).get("text", ""),vector_id) for vector_id, _ in results]

    def delete(self, vector_id: str) -> None:
        """
        将向量标记为无效，而不从 FAISS 索引中移除。
        """
        if vector_id in self.metadata:
            self.metadata[vector_id]['is_valid'] = False  # 标记为无效
            self.invalid_count += 1  # 增加无效向量计数器
            # 仅在无效计数超过某个批次阈值时，考虑重建索引
            if self.invalid_count >= 10:  # 例如每10次无效删除后检查一次
                self.rebuild_index()

    def update(self, vector_id: str, text: str, model: str = "text-embedding-3-small") -> None:
        """
        更新已有的向量和元数据。FAISS 没有直接的更新方法，因此需要先删除再插入。
        """
        if vector_id in self.metadata:
            self.delete(vector_id)  # 逻辑上删除旧的向量
            embedding = self.embedding_helper.get_embedding_with_cache(text, model=model)
            self._insert_vector(vector_id, embedding, metadata={"text": text})
        else:
            raise KeyError(f"Vector ID '{vector_id}' 不存在，无法更新。")

    def clear(self) -> None:
        """
        清空数据库，重置索引和元数据。
        """
        self.index.reset()  # FAISS 索引重置
        self.metadata.clear()
        self.vector_id_list.clear()  # 重置 vector_id_list 以保持一致

    def rebuild_index(self) -> None:
        """
        重新构建 FAISS 索引，去除标记为无效的向量。
        """
        # 如果无效向量占比超过设定的阈值，则重建索引
        total_vectors = len(self.metadata)
        if total_vectors == 0:
            return  # 数据库为空时不执行重建
        invalid_ratio = self.invalid_count / total_vectors
        if invalid_ratio <= self.rebuild_threshold:
            return
        valid_vectors = []
        valid_metadata = []
        new_vector_id_list = []  # 用于存储有效的 vector_id 列表

        for vector_id, metadata in self.metadata.items():
            if metadata.get('is_valid', True):
                valid_vectors.append(self.vectors[vector_id])  # 从 vectors 中获取有效向量
                valid_metadata.append(metadata)
                new_vector_id_list.append(vector_id)  # 记录有效的 vector_id
        # 将有效向量转换为 (n, d) 形式的矩阵
        valid_vectors_np = np.vstack(valid_vectors)

        # 计算有效向量的数量 n
        n = valid_vectors_np.shape[0]
        # 清空当前索引
        self.index.reset()
        self.metadata = {}
        self.vector_id_list.clear()  # 清空 vector_id_list

        # 重新插入有效向量 (n, vectors_np)
        self.index.add(n, valid_vectors_np)

        # 更新 metadata 和 vector_id_list
        for vector_id, metadata in zip(new_vector_id_list, valid_metadata):
            self.metadata[vector_id] = metadata
            self.vector_id_list.append(vector_id)
        # 重置无效计数器，因为索引已经重建
        self.invalid_count = 0


import numpy as np
from typing import List, Tuple, Optional


class InMemoryVectorDatabase(VectorDatabase):

    def __init__(self, n_components: int = None):
        self.vectors = {}
        self.metadata = {}
        self.embedding_helper = EmbeddingHelper()

    def _insert_vector(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        self.vectors[vector_id] = np.array(vector)
        if metadata:
            self.metadata[vector_id] = metadata

    def insert(self, text: str, model: str = "text-embedding-3-small") -> None:
        """
        插入单个文本的嵌入
        """
        vector_id = str(uuid.uuid4())  # 使用 UUID 生成唯一 ID
        embedding = self.embedding_helper.get_embedding_with_cache(text, model=model)  # 获取文本的嵌入
        self._insert_vector(vector_id, embedding, metadata={"text": text})

    def insert_batch(self, texts: List[str], model: str = "text-embedding-3-small"):
        """
        批量插入多个文本的嵌入
        """
        embeddings = self.embedding_helper.get_embeddings(texts, model=model)  # 获取多个文本的嵌入
        vectors_np = np.array(embeddings)
        for text, embedding in zip(texts, embeddings):
            vector_id = str(uuid.uuid4())  # 使用 UUID 生成唯一 ID
            self._insert_vector(vector_id, embedding, metadata={"text": text})

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        根据查询文本生成嵌入并搜索最相似的 top_k 个向量
        """
        all_text = [value.get("text") for value in self.metadata.values()]
        return self.embedding_helper.get_most_similar_strings(all_text, query, top_k)

    def delete(self, vector_id: str) -> None:
        """
        删除指定的向量及其元数据
        """
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            if vector_id in self.metadata:
                del self.metadata[vector_id]

    def update(self, vector_id: str, text: str, model: str = "text-embedding-3-small") -> None:
        """
        更新已有的向量及其元数据
        """
        if vector_id in self.vectors:
            self.delete(vector_id)
            embedding = self.embedding_helper.get_embedding_with_cache(text, model=model)
            self._insert_vector(vector_id, embedding, metadata={"text": text})
        else:
            raise KeyError(f"Vector ID '{vector_id}' 不存在，无法更新。")

    def clear(self) -> None:
        """
        清除所有存储的向量和元数据
        """
        self.vectors.clear()
        self.metadata.clear()


class VectorDatabaseFactory:
    @staticmethod
    def get_database(db_type: str, dimension: int = 1536) -> VectorDatabase:
        if db_type == 'memory':
            return InMemoryVectorDatabase()
        elif db_type == 'faiss':
            return FAISSVectorDatabase(dimension)
        # 后续可以扩展实际数据库实现，如 Pinecone 或 Milvus
        raise ValueError(f"Unknown database type: {db_type}")
