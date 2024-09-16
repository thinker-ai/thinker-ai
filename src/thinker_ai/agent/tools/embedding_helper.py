import pickle
from enum import Enum
from typing import cast, List, Tuple, Union

from numpy._typing import NDArray
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from thinker_ai.agent.provider import OpenAILLM
from thinker_ai.configs.config import config
from thinker_ai.configs.llm_config import LLMType
from thinker_ai.context_mixin import ContextMixin
from thinker_ai.configs.const import PROJECT_ROOT
import pandas as pd
import numpy as np


class SimilarityMetric(Enum):
    """
    枚举类型，表示支持的相似度和距离计算算法。
    """
    COSINE = ("cosine",
              "余弦相似度，适合语义嵌入的距离度量方式，因为它能够忽略向量的大小，专注于方向上的相似性，这与语义相似度非常吻合")
    EUCLIDEAN = (
        "euclidean", "欧几里得距离，两个向量在空间中的绝对距离（即线性距离），它考虑了向量的大小，这点和余弦相似度不同")
    MANHATTAN = ("manhattan", "曼哈顿距离，适用于在网格上只能水平或垂直移动的场景，计算的是各坐标轴差值的总和")
    CHEBYSHEV = ("chebyshev",
                 " 切比雪夫距离，适用于可以沿任何方向（水平、垂直或对角线）移动的场景，计算的是各坐标轴差值的最大值，这决定了移动的最小步数")

    def __init__(self, label: str, description: str):
        self.label = label
        self.description = description


class EmbeddingHelper:
    """
    一个处理嵌入向量获取、缓存和相似度计算的管理类
    """

    def __init__(self, embedding_cache_path: str = str(PROJECT_ROOT / "data/embeddings_cache.pkl")):
        """
        初始化嵌入管理器，加载缓存，并根据配置初始化 LLM 客户端。

        :param embedding_cache_path: 嵌入缓存文件路径
        """
        self.embedding_cache_path = embedding_cache_path
        # 如果缓存文件存在，则加载缓存
        try:
            self.embedding_cache = pd.read_pickle(self.embedding_cache_path)
        except FileNotFoundError:
            self.embedding_cache = {}

        # 根据配置初始化 LLM 客户端
        if config.llm.api_type == LLMType.OPENAI:
            self.client = (cast(OpenAILLM, ContextMixin().llm)).client
        else:
            raise NotImplementedError(f"不支持的 API 类型: {config.llm.api_type}")

    def _save_cache(self):
        """将缓存保存到磁盘。"""
        with open(self.embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(self.embedding_cache, embedding_cache_file)

    def _fetch_embedding(self, text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
        """
        使用同步 API 调用获取嵌入向量。

        :param text: 输入文本
        :param model: 要使用的嵌入模型
        :param kwargs: 额外的请求参数
        :return: 嵌入向量（浮点数列表）
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=model, **kwargs)
        return response.data[0].embedding

    async def _fetch_embedding_async(self, text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
        """
        使用异步 API 调用获取嵌入向量。

        :param text: 输入文本
        :param model: 要使用的嵌入模型
        :param kwargs: 额外的请求参数
        :return: 嵌入向量（浮点数列表）
        """
        text = text.replace("\n", " ")
        response = await self.client.embeddings.create(input=[text], model=model, **kwargs)
        return response.data[0].embedding

    def get_embedding_with_cache(self, string: str, model="text-embedding-3-small") -> List[float]:
        """
        从缓存中获取嵌入向量，如果缓存中没有则通过 API 获取并存入缓存。

        :param string: 要获取嵌入的文本
        :param model: 要使用的嵌入模型
        :return: 嵌入向量
        """
        if (string, model) not in self.embedding_cache:
            self.embedding_cache[(string, model)] = self.get_embedding_without_cache(string, model)
            self._save_cache()
        return self.embedding_cache[(string, model)]

    async def aget_embedding_with_cache(self, string: str, model="text-embedding-3-small") -> List[float]:
        """
        异步方式从缓存中获取嵌入向量，如果缓存中没有则通过 API 获取并存入缓存。

        :param string: 要获取嵌入的文本
        :param model: 要使用的嵌入模型
        :return: 嵌入向量
        """
        if (string, model) not in self.embedding_cache:
            self.embedding_cache[(string, model)] = await self.aget_embedding_without_cache(string, model)
            self._save_cache()
        return self.embedding_cache[(string, model)]

    def get_embedding_without_cache(self, string: str, model="text-embedding-3-small") -> List[float]:
        """
        从缓存中获取嵌入向量，如果缓存中没有则通过 API 获取并存入缓存。

        :param string: 要获取嵌入的文本
        :param model: 要使用的嵌入模型
        :return: 嵌入向量
        """
        return self._fetch_embedding(string, model)

    async def aget_embedding_without_cache(self, string: str, model="text-embedding-3-small") -> List[float]:
        """
        异步方式从缓存中获取嵌入向量，如果缓存中没有则通过 API 获取并存入缓存。

        :param string: 要获取嵌入的文本
        :param model: 要使用的嵌入模型
        :return: 嵌入向量
        """
        return await self._fetch_embedding_async(string, model)

    def is_related(self, text1: str, text2: str, similarity_threshold: float) -> bool:
        """
        判断两个文本在记忆网络中的相关性，通过余弦相似度判断。
        :param text1: 第一个文本。
        :param text2: 第二个文本。
        :return: 如果两个文本相关，返回 True，否则返回 False。
        """
        embedding1 = self.get_embedding_with_cache(text1)
        embedding2 = self.get_embedding_with_cache(text2)

        # 计算两个文本的余弦相似度
        similarity = cosine_similarity(embedding1, embedding2)

        # 设定相似度阈值
        return similarity > similarity_threshold

    def get_embeddings(self, texts: List[str], model="text-embedding-3-small", **kwargs) -> List[List[float]]:
        """
        获取多个文本的嵌入向量。

        :param texts: 文本列表
        :param model: 要使用的嵌入模型
        :param kwargs: 额外的请求参数
        :return: 每个文本的嵌入向量列表
        """
        assert len(texts) <= 2048, "批量大小不能超过 2048。"
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=model, **kwargs)
        return [item.embedding for item in response.data]

    async def aget_embeddings(self, texts: List[str], model="text-embedding-3-small", **kwargs) -> List[List[float]]:
        """
        异步方式获取多个文本的嵌入向量。

        :param texts: 文本列表
        :param model: 要使用的嵌入模型
        :param kwargs: 额外的请求参数
        :return: 每个文本的嵌入向量列表
        """
        assert len(texts) <= 2048, "批量大小不能超过 2048。"
        texts = [text.replace("\n", " ") for text in texts]
        response = await self.client.embeddings.create(input=texts, model=model, **kwargs)
        return [item.embedding for item in response.data]

    def get_most_similar_strings(self, source_strings: List[str], compare_string: str, k: int = 1,
                                 embedding_model="text-embedding-3-small",
                                 metric: SimilarityMetric = SimilarityMetric.COSINE) -> List[Tuple[str, float]]:
        """
        返回给定字符串的前k个最相似的字符串及其相似度。

        :param source_strings: 源字符串列表
        :param compare_string: 用于比较的字符串
        :param k: 返回的相似字符串数量
        :param embedding_model: 使用的嵌入模型
        :param metric: 相似度度量方式
        :return: 返回前k个最相似的字符串及其相似度
        """
        # 获取比较字符串的嵌入
        compare_embedding = self.get_embedding_with_cache(compare_string, model=embedding_model)

        # 获取所有源字符串的嵌入
        source_embeddings = [self.get_embedding_with_cache(source_string, model=embedding_model) for source_string in
                             source_strings]

        # 使用 sorted_by_similar_np 对嵌入进行排序
        sorted_results = sort_by_similar_np(compare_embedding, source_embeddings, metric=metric, desc=True)

        # 获取 k 个最相似的字符串
        return [(source_strings[source_embeddings.index(sorted_embedding)], similarity)
                for sorted_embedding, similarity in sorted_results[:k]]

    def get_most_similar_strings_by_index(self, source_strings: List[str], compare_string_index: int, k: int = 1,
                                          embedding_model="text-embedding-3-small") -> List[Tuple[str, float]]:
        """
        通过索引来获取与某个源字符串最相似的字符串。

        :param source_strings: 源字符串列表
        :param compare_string_index: 用于比较的源字符串索引
        :param k: 返回的相似字符串数量
        :param embedding_model: 使用的嵌入模型
        :return: 返回前k个最相似的字符串及其相似度
        """
        return self.get_most_similar_strings(source_strings, source_strings[compare_string_index], k, embedding_model)


def compute_similarity(query_vector: NDArray[np.float64], vector_matrix: NDArray[np.float64],
                       metric: SimilarityMetric = SimilarityMetric.COSINE) -> np.ndarray:
    """
    计算查询向量与向量矩阵之间的相似度或距离。

    :param query_vector: 待比较的单个向量 (1, d)
    :param vector_matrix: 多个嵌入向量 (n, d)
    :param metric: 使用的相似度或距离度量（枚举类型）
    :return: 相似度或距离数组，大小为 (n,)
    """
    # 如果 query_vector 是一维的，自动 reshape 为二维 (1, d)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # 如果 vector_matrix 不是二维的，自动 reshape 或进行其他处理
    if vector_matrix.ndim == 1:
        # 如果是 1 维，将其变成 2 维数组
        vector_matrix = vector_matrix.reshape(1, -1)
    elif vector_matrix.ndim > 2:
        raise ValueError("vector_matrix 必须是小于2的数组")

    if metric == SimilarityMetric.COSINE:
        return cosine_similarity(query_vector, vector_matrix)[0]  # 余弦相似度
    elif metric == SimilarityMetric.EUCLIDEAN:
        return -euclidean_distances(query_vector, vector_matrix)[0]  # 负的欧氏距离
    elif metric == SimilarityMetric.MANHATTAN:
        return -manhattan_distances(query_vector, vector_matrix)[0]  # 负的曼哈顿距离
    elif metric == SimilarityMetric.CHEBYSHEV:
        return -np.array([distance.chebyshev(query_vector, vector) for vector in vector_matrix])  # 切比雪夫距离
    else:
        raise ValueError(f"Unsupported similarity metric: {metric.label}")


def distances_from_embeddings(query_embedding: List[float], embeddings: List[List[float]],
                              metric: SimilarityMetric) -> List[float]:
    """
    根据给定的距离度量计算查询向量与多个嵌入向量之间的距离。

    :param query_embedding: 待比较的单个向量
    :param embeddings: 多个嵌入向量
    :param metric: 使用的距离度量 (枚举类型)
    :return: 一维的距离数组
    """
    query_embedding_np = np.array(query_embedding)
    embeddings_np = np.array(embeddings)

    # 计算相似度或距离
    distances = compute_similarity(query_embedding_np, embeddings_np, metric)

    # 返回结果，保持一致的接口（距离为正值），并返回一维数组
    return np.abs(distances).tolist()


def sort_by_metric(metric_values: Union[List[float], np.ndarray], desc: bool = True) -> np.ndarray:
    """
    根据相似度或距离值进行排序，返回排序后的索引。

    :param metric_values: 相似度或距离的值列表
    :param desc: 是否按降序排序
    :return: 排序后的索引
    """
    return np.argsort(metric_values)[::-1] if desc else np.argsort(metric_values)


def sort_by_similar_np(compare_embedding: List[float], embeddings: List[List[float]],
                       metric: SimilarityMetric = SimilarityMetric.COSINE, desc: bool = True) -> List[
    Tuple[List[float], float]]:
    """
    根据相似度度量对嵌入进行排序，返回排序后的嵌入和相似度。

    :param compare_embedding: 要比较的向量 (1, d)
    :param embeddings: 多个嵌入向量 (n, d)
    :param metric: 使用的相似度度量 (SimilarityMetric 枚举类型)
    :param desc: 是否按降序排序
    :return: 排序后的嵌入和相似度
    """
    similarities: List[float] = distances_from_embeddings(compare_embedding, embeddings, metric)

    # 使用统一的排序函数
    sorted_indices = sort_by_metric(similarities, desc)

    # 根据排序后的索引获取排序后的嵌入向量和相似度
    sorted_embeddings = np.array(embeddings)[sorted_indices]
    sorted_similarities = np.array(similarities)[sorted_indices]

    # 将嵌入向量和相似度打包成元组列表返回
    sorted_results = list(zip(sorted_embeddings.tolist(), sorted_similarities.tolist()))
    return sorted_results


def pca_components_from_embeddings(embeddings: List[List[float]], n_components=2) -> np.ndarray:
    """
    使用PCA降维算法对嵌入进行降维。

    :param embeddings: 要降维的嵌入向量列表
    :param n_components: 降维后的维度数量
    :return: PCA后的嵌入矩阵
    """
    pca = PCA(n_components=n_components)
    array_of_embeddings = np.array(embeddings)
    return pca.fit_transform(array_of_embeddings)


def tsne_components_from_embeddings(embeddings: List[List[float]], n_components=2, **kwargs) -> np.ndarray:
    """
    使用t-SNE算法对嵌入进行降维。

    :param embeddings: 要降维的嵌入向量列表
    :param n_components: 降维后的维度数量
    :param kwargs: 其他传递给t-SNE的参数
    :return: t-SNE降维后的嵌入矩阵
    """
    if "init" not in kwargs.keys():
        kwargs["init"] = "pca"
    if "learning_rate" not in kwargs.keys():
        kwargs["learning_rate"] = "auto"
    tsne = TSNE(n_components=n_components, **kwargs)
    array_of_embeddings = np.array(embeddings)
    return tsne.fit_transform(array_of_embeddings)


def indices_of_nearest_neighbors_from_distances(distances: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    返回根据距离排序后的最近邻索引列表。
    :param distances: 距离列表
    :return: 索引列表
    """
    return np.argsort(distances)
