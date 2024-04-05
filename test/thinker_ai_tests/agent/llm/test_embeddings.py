from typing import Optional

import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure

from thinker_ai.agent.llm.embeddings import get_most_similar_strings_by_index, embedding_from_string, \
    tsne_components_from_embeddings, chart_from_components, plot_multiclass_precision_recall
from thinker_ai.context import get_project_root


class TestEmbeddingFunctions(unittest.TestCase):
    df: Optional[pd.DataFrame] = None
    n_examples: int = 5

    @classmethod
    def setUpClass(cls):
        """Load test data from the CSV file."""
        dataset_path = "data/AG_news_samples.csv"
        cls.df = pd.read_csv(dataset_path)
        cls.df.head(cls.n_examples)
        for idx, row in cls.df.head(cls.n_examples).iterrows():
            print("")
            print(f"Title: {row['title']}")
            print(f"Description: {row['description']}")
            print(f"Label: {row['label']}")

    def test_get_most_similar_strings_by_index(self):
        # 使用前5个样本进行测试
        for idx in range(self.n_examples):
            title = self.df.iloc[idx]['title']
            description = self.df.iloc[idx]['description']
            label = self.df.iloc[idx]['label']
            print(f"\nTesting with Title: {title}\nDescription: {description}\nLabel: {label}")

            # 获取最相似的字符串
            source_strings = self.df['description'].tolist()[:self.n_examples]
            similar_strings = get_most_similar_strings_by_index(source_strings, idx, k=1,
                                                                embedding_model="text-embedding-3-small")

            # 确保返回的是字符串列表
            self.assertIsInstance(similar_strings, list)
            self.assertEqual(len(similar_strings), 1)
            print(f"Most similar string to \"{description[:30]}...\": {similar_strings[0][:30]}...")

    def test_tsne_components_from_embeddings(self):
        # get embeddings for all article descriptions
        article_descriptions = self.df["description"].head(5).tolist()
        embeddings = [embedding_from_string(string) for string in article_descriptions]
        # compress the 2048-dimensional embeddings into 2 dimensions using t-SNE
        tsne_components = tsne_components_from_embeddings(embeddings, perplexity=4)
        # get the article labels for coloring the chart
        labels = self.df["label"].head(5).tolist()

        figure: Figure = chart_from_components(
            components=tsne_components,
            labels=labels,
            strings=article_descriptions,
            width=600,
            height=500,
            title="t-SNE components of article descriptions",
        )
        figure.write_image("data/tsne_test_output.png")
        self.assertIsInstance(figure,Figure)


    def test_plot_multiclass_precision_recall(self):
        # 创建假的预测得分和真实标签
        y_true_untransformed = np.array([1, 2, 1, 2, 0])
        y_score = np.array([[0.5, 0.2, 0.3],
                            [0.1, 0.8, 0.1],
                            [0.7, 0.1, 0.2],
                            [0.2, 0.6, 0.2],
                            [0.2, 0.2, 0.6]])
        class_list = [0, 1, 2]
        classifier_name = "Test Classifier"

        # 绘制精确率-召回率曲线
        plot_multiclass_precision_recall(y_score, y_true_untransformed, class_list, classifier_name)
        plt.gcf().show()
        # 验证是否成功创建了图表
        self.assertIsInstance(plt.gcf(), plt.Figure)
        plt.close()  # 关闭绘制的图表，避免在测试中实际显示

if __name__ == "__main__":
    unittest.main()
