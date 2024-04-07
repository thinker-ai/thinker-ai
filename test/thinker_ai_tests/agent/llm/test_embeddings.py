from typing import Optional

import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure
from sklearn.metrics import f1_score

from thinker_ai.agent.llm.embeddings import get_most_similar_strings_by_index, get_embedding_with_cache, \
    tsne_components_from_embeddings, chart_from_components, plot_multiclass_precision_recall, \
    predict_with_zero_shot, predict_with_sample

EMBEDDING_MODEL = "text-embedding-3-small"

semantic_text = [
    'excellent product', 'horrible product', 'do not buy', 'highly recommended',
    'broke after a week', 'best purchase ever', 'worst purchase of my life', 'very satisfied',
    'completely dissatisfied', 'exceeded my expectations', 'back to the store', 'saving a lot of time',
    'not worth the price', 'happy with this purchase', 'regret buying it', 'on my top list',
    'failed the first time I used it', 'love it', 'hate it', 'not what I expected'
]
semantic_values = [
    'positive', 'negative', 'negative', 'positive',
    'negative', 'positive', 'positive', 'positive',
    'negative', 'positive', 'positive', 'positive',
    'negative', 'positive', 'negative', 'positive',
    'negative', 'positive', 'negative', 'negative'
]


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
        embeddings = [get_embedding_with_cache(string) for string in article_descriptions]
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
        self.assertIsInstance(figure, Figure)

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

    def test_predict_with_zero_shot(self):
        """
        测试 classify_with_zero_shot 函数。
        """
        # 待分类文本和标签
        labels_for = ['positive', 'negative']
        # 执行分类
        results_df = predict_with_zero_shot(
            to_predict_texts=semantic_text,
            labels_for=labels_for,
            confidence_threshold=0.1
        )
        # 验证是否每个文本都有预测结果
        self.assertEqual(len(results_df), len(semantic_text))
        # 验证结果是否为 DataFrame
        self.assertIn('text', results_df.columns)
        self.assertIn('predicted_label', results_df.columns)
        self.assertIn('confidence', results_df.columns)

        self.assertTrue(all(results_df['confidence'] >= 0))
        self.assertTrue(all(results_df['confidence'] <= 1))
        for confidence in results_df['confidence']:
            print(confidence)
        # 计算F1分数
        f1 = f1_score(semantic_values, results_df['predicted_label'].tolist(), average='macro')
        # 检查F1分数是否达到预期水平
        self.assertGreater(f1, 0.8, "F1分数未达到预期水平")

    def test_predict_from_samples(self):
        """
        测试预测质量高于阈值时的情况。
        """
        # 模拟数据
        samples_data = {
            'text_column':semantic_text,
            'Semantic': semantic_values
        }
        samples_df = pd.DataFrame(samples_data)

        target_col = 'Semantic'
        to_predict_texts = [
            'utterly disappointing experience', 'exceedingly pleased with the quality', 'would not recommend to anyone',
            'a must-have for everyone',
            'fell apart on first use', 'absolutely love this purchase', 'a total letdown',
            'went beyond my expectations',
            'not up to the mark', 'thrilled with the performance', 'returning it immediately',
            'makes life so much easier',
            'overpriced for what it offers', 'satisfaction guaranteed', 'waste of money', 'top of the line product',
            'malfunctioned after a few uses', 'a joy to use daily', 'deeply unsatisfying',
            'did not meet my expectations'
        ]
        expected_labels = [
            'negative',  # 'utterly disappointing experience'
            'positive',  # 'exceedingly pleased with the quality'
            'negative',  # 'wouldn't recommend to anyone'
            'positive',  # 'a must-have for everyone'
            'negative',  # 'fell apart on first use'
            'positive',  # 'absolutely love this purchase'
            'negative',  # 'a total letdown'
            'positive',  # 'went beyond my expectations'
            'negative',  # 'not up to the mark'
            'positive',  # 'thrilled with the performance'
            'negative',  # 'returning it immediately'
            'positive',  # 'makes life so much easier'
            'negative',  # 'overpriced for what it offers'
            'positive',  # 'satisfaction guaranteed'
            'negative',  # 'waste of money'
            'positive',  # 'top of the line product'
            'negative',  # 'malfunctioned after a few uses'
            'positive',  # 'a joy to use daily'
            'negative',  # 'deeply unsatisfying'
            'negative'  # 'did not meet my expectations'
        ]

        # 调用待测试的函数
        results_df = predict_with_sample(
            samples_df=samples_df,
            samples_target_col=target_col,
            to_predict_texts=to_predict_texts,
            embedding_model_name="text-embedding-3-small",
            quality_threshold=0.5  # 为了让测试正常，将低数据质量要求
        )

        # 验证结果的DataFrame结构
        self.assertEqual(len(results_df), len(to_predict_texts))
        self.assertTrue('text' in results_df.columns)
        self.assertTrue('predicted' in results_df.columns)
        self.assertTrue('confidence' in results_df.columns)

        # 验证预测结果的置信度，这里的具体值会根据你的实际模型和数据而有所不同
        # 这里仅做结构上的验证
        self.assertTrue(all(results_df['confidence'] >= 0))
        self.assertTrue(all(results_df['confidence'] <= 1))
        for confidence in results_df['confidence']:
            print(confidence)
        # 计算F1分数
        f1 = f1_score(expected_labels, results_df['predicted'].tolist(), average='macro')
        # 检查F1分数是否达到预期水平
        self.assertGreater(f1, 0.9, "F1分数未达到预期水平")


if __name__ == "__main__":
    unittest.main()
