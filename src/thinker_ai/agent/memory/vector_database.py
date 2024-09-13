from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class VectorDatabase(ABC):
    @abstractmethod
    def insert(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        """插入向量数据"""
        pass

    @abstractmethod
    def search(self, vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """检索与查询向量最相似的 top_k 向量"""
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> None:
        """删除指定的向量"""
        pass

    @abstractmethod
    def update(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        """更新已有的向量"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清除数据库中的所有向量"""
        pass


import faiss
import numpy as np
from typing import List, Tuple, Optional


class FAISSVectorDatabase(VectorDatabase):
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.vectors = {}
        self.metadata = {}

    def insert(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        vector_np = np.array(vector).astype('float32').reshape(1, -1)
        self.index.add(vector_np)
        self.vectors[vector_id] = vector_np
        if metadata:
            self.metadata[vector_id] = metadata

    def search(self, vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if len(self.vectors) == 0:
            return []

        query_vector = np.array(vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                vector_id = list(self.vectors.keys())[idx]
                results.append((vector_id, dist))
        return results

    def delete(self, vector_id: str) -> None:
        # FAISS 本身不支持单个删除操作，可以在重新构建索引时去除
        pass

    def update(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        self.insert(vector_id, vector, metadata)

    def clear(self) -> None:
        self.index.reset()
        self.vectors.clear()
        self.metadata.clear()


import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class InMemoryVectorDatabase(VectorDatabase):
    def __init__(self):
        self.vectors = {}
        self.metadata = {}

    def insert(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        self.vectors[vector_id] = np.array(vector)
        if metadata:
            self.metadata[vector_id] = metadata

    def search(self, vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []

        query_vector = np.array(vector).reshape(1, -1)
        vector_ids, vectors = zip(*self.vectors.items())
        vector_matrix = np.vstack(vectors)

        similarities = cosine_similarity(query_vector, vector_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(vector_ids[i], similarities[i]) for i in top_indices]

    def delete(self, vector_id: str) -> None:
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            if vector_id in self.metadata:
                del self.metadata[vector_id]

    def update(self, vector_id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        self.insert(vector_id, vector, metadata)

    def clear(self) -> None:
        self.vectors.clear()
        self.metadata.clear()


class VectorDatabaseFactory:
    @staticmethod
    def get_database(db_type: str, dimension: int = 128) -> VectorDatabase:
        if db_type == 'memory':
            return InMemoryVectorDatabase()
        elif db_type == 'faiss':
            return FAISSVectorDatabase(dimension)
        # 后续可以扩展实际数据库实现，如 Pinecone 或 Milvus
        raise ValueError(f"Unknown database type: {db_type}")
