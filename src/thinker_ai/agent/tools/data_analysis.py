from typing import List
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from thinker_ai.agent.tools.data_chart import plot_precision_recall
from thinker_ai.agent.tools.embedding_helper import EmbeddingHelper, SimilarityMetric, compute_similarity

embedding_helper = EmbeddingHelper()


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model: BaseEstimator) -> float:
    """
    训练给定模型并评估其质量，返回模型的 F1 质量分数。

    :param X_train: 训练集嵌入向量
    :param y_train: 训练集标签
    :param X_test: 测试集嵌入向量
    :param y_test: 测试集标签
    :param model: 可训练的模型 (如 RandomForestClassifier)
    :return: 测试集上的 F1 分数
    """
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    quality = f1_score(y_test, preds_test, average='macro')
    return quality


def predict_with_sample(
        samples_df: pd.DataFrame,
        samples_target_col: str,
        to_predict_texts: List[str],
        embedding_model_name: str = "text-embedding-3-small",
        quality_threshold: float = 0.5,
        model: BaseEstimator = RandomForestClassifier(random_state=42)  # 默认使用随机森林模型
) -> pd.DataFrame:
    """
    带有预测质量检查的通用预测方法。

    参数:
    - samples_df: DataFrame，包含特征和目标列。
    - samples_target_col: 目标列的名称。
    - to_predict_texts: 待预测的文本列表。
    - embedding_model_name: 使用的嵌入模型名称。
    - quality_threshold: 预测质量的阈值。
    - model: 用于预测的模型，默认使用 RandomForestClassifier。

    返回:
    - 包含预测结果的 DataFrame，包含每个预测数据的置信度信息。

    抛出:
    - ValueError: 如果模型的预测质量未达到预设阈值。
    """
    # 获取样本嵌入向量
    samples_df['embedding'] = samples_df['text_column'].apply(
        lambda x: embedding_helper.get_embedding_with_cache(x, embedding_model_name))
    X = np.array(samples_df['embedding'].tolist())
    y = samples_df[samples_target_col].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练并验证模型质量
    quality = train_and_evaluate_model(X_train, y_train, X_test, y_test, model)
    if quality < quality_threshold:
        raise ValueError(f"Model quality below threshold: {quality} < {quality_threshold}")

    # 对待预测文本生成嵌入向量并进行预测
    to_predict_embeddings = np.array(
        [embedding_helper.get_embedding_with_cache(text, embedding_model_name) for text in to_predict_texts]
    )
    probas = model.predict_proba(to_predict_embeddings)

    # 获取最可能的类别和相应的置信度
    preds = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)

    # 创建包含预测结果和置信度的 DataFrame
    results_df = pd.DataFrame({
        'text': to_predict_texts,
        'predicted': [model.classes_[i] for i in preds],  # 将索引转换为类别标签
        'confidence': confidences
    })
    return results_df


label_embeddings_cache = {}


def predict_with_zero_shot(
        to_predict_texts: List[str],
        labels_for: List[str],
        embedding_model: str = "text-embedding-3-small",
        confidence_threshold: float = 0.0,
        metric: SimilarityMetric = SimilarityMetric.COSINE  # 添加支持不同相似度度量
) -> pd.DataFrame:
    """
    使用零样本学习对文本进行分类。
    """

    if not to_predict_texts or not labels_for:
        return pd.DataFrame({
            'text': to_predict_texts,
            'predicted_label': [],
            'confidence': []
        })

    # 缓存标签的嵌入，避免重复计算
    if embedding_model not in label_embeddings_cache:
        label_embeddings_cache[embedding_model] = {}

    label_embeddings = []
    for label in labels_for:
        if label not in label_embeddings_cache[embedding_model]:
            label_embeddings_cache[embedding_model][label] = embedding_helper.get_embedding_with_cache(label,
                                                                                                       embedding_model)
        label_embeddings.append(label_embeddings_cache[embedding_model][label])

    label_embeddings_np = np.array(label_embeddings)

    predictions = []
    confidences = []

    for text in to_predict_texts:
        text_embedding = embedding_helper.get_embedding_with_cache(text, embedding_model)
        similarities = compute_similarity(np.array([text_embedding]), label_embeddings_np,metric=metric)

        max_similarity_index = np.argmax(similarities)
        prediction = labels_for[max_similarity_index]
        confidence = similarities[max_similarity_index]

        if confidence < confidence_threshold:
            prediction = "uncertain"

        predictions.append(prediction)
        confidences.append(confidence)

    # 输出 DataFrame 包含预测结果、置信度和相似度
    results_df = pd.DataFrame({
        'text': to_predict_texts,
        'predicted_label': predictions,
        'confidence': confidences,
    })

    return results_df


def calculate_percentile_cosine_similarity(data: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    # 计算不同聚类之间整体关系的亲疏，注意：这种相似度比较只是关于文本语义的相似度比较，而不是因果关系的概率比较，
    # 比如在推荐系统中，我们可能想要了解用户的评论文本与产品描述文本之间的语义相似性，或者在客户服务中分析客户反馈
    # 与已知问题之间的相似度。然而，对于关心不同类别数据之间如何随时间变化而产生互动的情况，我们需要另一种方法，
    # 另外，目前算法会受不平衡的个体关系的影响较大，必要时要对个体的权重影响进行考量。
    embeddings = {
        category: data[category].apply(embedding_helper.get_embedding_with_cache).tolist()
        for category in categories
    }

    # Calculate cumulative cosine similarity for each unique pair of categories
    results = []
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i + 1:]:
            # Initialize the cumulative similarity and count
            cumulative_similarity = 0
            count = 0

            # Calculate similarity for each element in cat1 to each element in cat2
            for emb1 in embeddings[cat1]:
                for emb2 in embeddings[cat2]:
                    cumulative_similarity += cosine_similarity(emb1, emb2)
                    count += 1

            # Compute the average similarity for the category pair
            average_similarity = cumulative_similarity / count if count else 0
            results.append({'Category_Pair': f'{cat1} & {cat2}', 'Cosine_Similarity': average_similarity})

    # Create a DataFrame from the results
    similarity_df = pd.DataFrame(results)
    similarity_df['Percentile_Rank'] = similarity_df['Cosine_Similarity'].rank(pct=True)

    return similarity_df


def compute_similarity_between_categories(data: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """
    计算多个类别之间的余弦相似度，并返回排序后的 DataFrame。

    :param data: 包含类别文本的 DataFrame
    :param categories: 需要比较的类别列表
    :return: 包含类别对及其相似度的 DataFrame，按相似度排序
    """
    # 获取每个类别的平均嵌入向量，转为numpy数组
    embeddings = []
    for category in categories:
        # 对每个类别的所有文本计算嵌入
        category_embeddings = np.array([embedding_helper.get_embedding_with_cache(text) for text in data[category]])
        # 计算每个类别的平均嵌入向量，确保每个类别只有一个向量
        category_mean_embedding = np.mean(category_embeddings, axis=0)
        embeddings.append(category_mean_embedding)

    # 转换为 numpy 数组
    embeddings_np = np.array(embeddings)

    # 初始化一个空的列表来存储相似度结果
    similarity_results = []

    # 遍历每个类别并计算它与其他类别的相似度，确保不重复
    for i, query_embedding in enumerate(embeddings_np):
        similarities = compute_similarity(query_embedding.reshape(1, -1), embeddings_np)
        for j, similarity in enumerate(similarities):
            if i < j:  # 确保只计算不重复的类别对
                similarity_results.append({
                    'Category_Pair': f"{categories[i]} & {categories[j]}",
                    'Cosine_Similarity': similarity
                })
    # 将结果转为DataFrame
    similarity_df = pd.DataFrame(similarity_results)

    # 计算百分位数并添加到DataFrame
    similarity_df['Percentile_Rank'] = similarity_df['Cosine_Similarity'].rank(pct=True)

    # 对数据进行排序
    similarity_df_sorted = similarity_df.sort_values('Percentile_Rank', ascending=False)

    return similarity_df_sorted


def compute_precision_recall(y_true: np.ndarray, y_score: np.ndarray, class_list: list):
    """
    计算每个类别的 precision-recall 曲线和平均精度分数。

    :param y_true: 真实标签 (n_samples, n_classes)
    :param y_score: 每个类的预测得分 (n_samples, n_classes)
    :param class_list: 类别列表
    :return: precision, recall, average_precision 字典以及微平均的相关结果
    """
    n_classes = len(class_list)

    # 确保 y_true_untransformed 是一个 (n_samples, n_classes) 的 2D 数组

    precision = {}
    recall = {}
    average_precision = {}

    # 计算每个类别的 precision, recall, 和 average precision
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # 计算 micro-average 的 precision, recall
    precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    average_precision_micro = average_precision_score(y_true, y_score, average="micro")

    return precision, recall, average_precision, precision_micro, recall_micro, average_precision_micro


def plot_multiclass_precision_recall(y_true: np.ndarray, y_score: np.ndarray, class_list: list, classifier_name: str):
    """
    Precision-Recall plotting for a multiclass problem. It computes precision-recall values and plots them for each class.

    :param y_score: 每个类的预测得分 (n_samples, n_classes)
    :param y_true: 真实标签
    :param class_list: 类别列表
    :param classifier_name: 分类器名称
    """
    # 调用计算 precision 和 recall 的方法
    precision, recall, average_precision, precision_micro, recall_micro, average_precision_micro = compute_precision_recall(
        y_true, y_score, class_list
    )

    # 输出总体的 average precision score
    print(f"{classifier_name} - Average precision score over all classes: {average_precision_micro:.2f}")

    # 调用绘图方法
    plot_precision_recall(precision, recall, average_precision, precision_micro, recall_micro, average_precision_micro,
                          class_list, classifier_name)
