import unittest
from typing import Optional
import pandas as pd
import numpy as np
from plotly.graph_objs import Figure
from thinker_ai.agent.tools.data_chart import chart_from_components_2D
from thinker_ai.agent.tools.embedding_helper import EmbeddingHelper, tsne_components_from_embeddings, compute_similarity, SimilarityMetric


class TestEmbeddingFunctions(unittest.TestCase):
    embedding_helper = EmbeddingHelper()
    df: Optional[pd.DataFrame] = None
    n_examples: int = 5

    @classmethod
    def setUpClass(cls):
        """加载测试数据集"""
        dataset_path = "data/AG_news_samples.csv"
        cls.df = pd.read_csv(dataset_path)
        cls.df.head(cls.n_examples)
        for idx, row in cls.df.head(cls.n_examples).iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"Description: {row['description']}")
            print(f"Label: {row['label']}")

    def test_get_most_similar_strings_by_index(self):
        """测试get_most_similar_strings_by_index函数"""
        for idx in range(self.n_examples):
            title = self.df.iloc[idx]['title']
            description = self.df.iloc[idx]['description']
            label = self.df.iloc[idx]['label']
            print(f"\nTesting with Title: {title}\nDescription: {description}\nLabel: {label}")

            # 获取最相似的字符串
            source_strings = self.df['description'].tolist()[:self.n_examples]
            similar_strings = self.embedding_helper.get_most_similar_strings_by_index(source_strings, idx, k=1,
                                                                                     embedding_model="text-embedding-3-small")

            # 验证返回的字符串列表是否正确
            self.assertIsInstance(similar_strings, list)
            self.assertEqual(len(similar_strings), 1)
            print(f"Most similar string to \"{description[:30]}...\": {similar_strings[0][:30]}...")

    def test_tsne_components_from_embeddings(self):
        """测试t-SNE嵌入降维可视化"""
        article_descriptions = self.df["description"].head(5).tolist()
        embeddings = [self.embedding_helper.get_embedding_with_cache(string) for string in article_descriptions]
        tsne_components = tsne_components_from_embeddings(embeddings, perplexity=4)
        labels = self.df["label"].head(5).tolist()

        figure: Figure = chart_from_components_2D(
            components=tsne_components,
            labels=labels,
            strings=article_descriptions,
            width=600,
            height=500,
            title="t-SNE components of article descriptions",
        )
        figure.write_image("data/tsne_test_output.png")
        self.assertIsInstance(figure, Figure)

    def test_get_embedding_with_cache(self):
        """测试嵌入获取功能及缓存机制"""
        sample_text = "This is a test sentence."
        embedding = self.embedding_helper.get_embedding_with_cache(sample_text, model="text-embedding-3-small")
        self.assertIsInstance(embedding, list)
        self.assertTrue(len(embedding) > 0)

        # 验证缓存是否有效
        cached_embedding = self.embedding_helper.get_embedding_with_cache(sample_text, model="text-embedding-3-small")
        self.assertEqual(embedding, cached_embedding)

    async def test_async_embedding_with_cache(self):
        """异步测试嵌入获取功能"""
        sample_text = "Async test sentence."
        embedding = await self.embedding_helper.aget_embedding_with_cache(sample_text, model="text-embedding-3-small")
        self.assertIsInstance(embedding, list)
        self.assertTrue(len(embedding) > 0)

    def test_compute_similarity(self):
        """测试compute_similarity函数的不同度量方式"""
        vector1 = np.random.rand(1, 1536)  # 随机生成向量1
        vector2 = np.random.rand(5, 1536)  # 随机生成向量2

        # 测试余弦相似度
        cosine_sim = compute_similarity(vector1, vector2, metric=SimilarityMetric.COSINE)
        self.assertIsInstance(cosine_sim, np.ndarray)
        self.assertEqual(len(cosine_sim), 5)

        # 测试欧几里得距离
        euclidean_sim = compute_similarity(vector1, vector2, metric=SimilarityMetric.EUCLIDEAN)
        self.assertIsInstance(euclidean_sim, np.ndarray)
        self.assertEqual(len(euclidean_sim), 5)

        # 测试曼哈顿距离
        manhattan_sim = compute_similarity(vector1, vector2, metric=SimilarityMetric.MANHATTAN)
        self.assertIsInstance(manhattan_sim, np.ndarray)
        self.assertEqual(len(manhattan_sim), 5)

    def test_compute_similarity_invalid_input(self):
        """测试compute_similarity的容错处理"""
        vector1 = np.random.rand(1536)  # 生成一维向量
        vector2 = np.random.rand(1536)  # 生成一维向量

        # 尝试计算相似度，不应该抛出异常
        try:
            result = compute_similarity(vector1, vector2)
            self.assertTrue(True, "计算相似度时没有抛出异常")
        except ValueError as e:
            self.fail(f"计算相似度时不应该抛出异常，错误: {str(e)}")

    def test_get_most_similar_strings_error_handling(self):
        """测试get_most_similar_strings函数的错误处理"""
        source_strings = ["Test string 1", "Test string 2"]
        compare_string = "Another test string"

        try:
            similar_strings = self.embedding_helper.get_most_similar_strings(source_strings, compare_string, k=1,
                                                                             embedding_model="text-embedding-3-small")
            self.assertTrue(True, "计算最相似字符串时没有抛出异常")
        except Exception as e:
            self.fail(f"计算最相似字符串时不应该抛出异常，错误: {str(e)}")


if __name__ == "__main__":
    unittest.main()