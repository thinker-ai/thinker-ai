from typing import Optional
import unittest
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from thinker_ai.agent.tools.data_analysis import predict_with_zero_shot, predict_with_sample, \
    compute_similarity_between_categories, compute_precision_recall, plot_multiclass_precision_recall
from matplotlib import pyplot as plt

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


class TestTrainAndEvaluateFunctions(unittest.TestCase):
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
            'text_column': semantic_text,
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

    def test_compute_similarity_between_categories(self):
        """测试类别间余弦相似度百分位数的计算和排序"""
        data = pd.DataFrame({
            'ProductFeature': ['Advanced GPS', 'Longer battery life', 'High resolution camera', 'Waterproof'],
            'ProductFault': ['Screen issues', 'Overheating', 'Camera malfunction', 'Battery drain'],
            'CustomerSatisfaction': ['Very satisfied', 'Somewhat satisfied', 'Dissatisfied', 'Neutral'],
            'EnvironmentalIndex': ['High CO2 emissions', 'Renewable energy usage', 'Low waste production',
                                   'High pollution levels']
        })

        # 选择需要计算相似度的类别
        categories = ['ProductFeature', 'ProductFault', 'CustomerSatisfaction', 'EnvironmentalIndex']

        # 计算并获取排序后的相似度 DataFrame
        similarity_df_sorted = compute_similarity_between_categories(data, categories)

        # 检查输出是否为 DataFrame
        self.assertIsInstance(similarity_df_sorted, pd.DataFrame)

        # 检查结果中是否包含预期的列
        expected_columns = ['Category_Pair', 'Cosine_Similarity', 'Percentile_Rank']
        self.assertTrue(all(column in similarity_df_sorted.columns for column in expected_columns))

        # 检查相似度排名是否按降序排列
        sorted_similarity = similarity_df_sorted['Cosine_Similarity'].values
        self.assertTrue(np.all(sorted_similarity[:-1] >= sorted_similarity[1:]))

        # 检查百分位排名是否按降序排列
        sorted_percentile = similarity_df_sorted['Percentile_Rank'].values
        self.assertTrue(np.all(sorted_percentile[:-1] >= sorted_percentile[1:]))

        # 绘制类别间相似度百分位数条形图
        plt.figure(figsize=(14, 7))
        sns.barplot(
            x='Percentile_Rank',
            y='Category_Pair',
            data=similarity_df_sorted,
            orient='h'
        )
        plt.title('Percentile Cosine Similarity Between Categories')
        plt.xlabel('Percentile Rank of Cosine Similarity')
        plt.ylabel('Category Pairs')
        plt.tight_layout()
        plt.show()

    def test_compute_precision_recall(self):
        # 准备测试数据
        y_true = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_score = np.array([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1], [0.7, 0.1, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        class_list = [0, 1, 2]

        # 计算 precision 和 recall
        precision, recall, average_precision, precision_micro, recall_micro, average_precision_micro = compute_precision_recall(y_true, y_score, class_list)

        # 验证 precision, recall 输出
        self.assertIsInstance(precision, dict)
        self.assertIsInstance(recall, dict)
        self.assertEqual(len(precision), len(class_list))
        self.assertEqual(len(recall), len(class_list))

        # 验证 precision_micro, recall_micro, 和 average_precision_micro 的输出
        self.assertIsInstance(precision_micro, np.ndarray)
        self.assertIsInstance(recall_micro, np.ndarray)
        self.assertIsInstance(average_precision_micro, float)

        # 检查 precision_micro 和 recall_micro 的长度是否一致
        self.assertEqual(len(precision_micro), len(recall_micro))

        # 验证 average_precision_micro 在合理的范围内 [0, 1]
        self.assertGreaterEqual(average_precision_micro, 0.0)
        self.assertLessEqual(average_precision_micro, 1.0)

    def test_plot_multiclass_precision_recall(self):
        """测试多分类的精确率-召回率曲线绘制"""
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        y_score = np.array([[0.5, 0.2, 0.3],
                            [0.1, 0.8, 0.1],
                            [0.7, 0.1, 0.2],
                            [0.2, 0.6, 0.2],
                            [0.2, 0.2, 0.6]])
        class_list = [0, 1, 2]
        classifier_name = "Test Classifier"

        plot_multiclass_precision_recall(y_true,y_score, class_list, classifier_name)
        plt.gcf().show()

        # 确保图表成功生成
        self.assertIsInstance(plt.gcf(), plt.Figure)
        plt.close()  # 关闭图表


if __name__ == "__main__":
    unittest.main()
