import pickle
import textwrap as tr
from typing import List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import plotly.express as px
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, silhouette_score

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from thinker_ai.agent.llm import gpt
from thinker_ai.context import get_project_root
from typing import List
import pandas as pd

client = gpt.llm


def get_embedding_without_cache(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


async def aget_embedding(
        text: str, model="text-embedding-3-small", **kwargs
) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return (await client.embeddings.create(input=[text], model=model, **kwargs))[
        "data"
    ][0]["embedding"]


def get_embeddings(
        list_of_text: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = client.embeddings.create(input=list_of_text, model=model, **kwargs).data
    return [d.embedding for d in data]


async def aget_embeddings(
        list_of_text: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = (
        await client.embeddings.create(input=list_of_text, model=model, **kwargs)
    ).data
    return [d.embedding for d in data]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_np(compare_embedding_np, embeddings_np) -> float:
    dot_products = np.dot(embeddings_np, compare_embedding_np)
    norms = np.linalg.norm(embeddings_np, axis=1) * np.linalg.norm(compare_embedding_np)
    similarities = dot_products / norms
    return similarities


def sorted_by_similar_np(compare_embedding: List[float], embeddings: List[List[float]], desc: bool = True) -> List[
    Tuple[List[float], float]]:
    compare_embedding_np = np.array(compare_embedding)
    embeddings_np = np.array(embeddings)

    # 向量化计算余弦相似度
    similarities = cosine_similarity_np(compare_embedding_np, embeddings_np)

    # 获取排序后的索引
    sorted_indices = np.argsort(similarities)[::-1] if desc else np.argsort(similarities)

    # 根据索引排序嵌入向量和相似度
    sorted_embeddings = embeddings_np[sorted_indices]
    sorted_similarities = similarities[sorted_indices]

    # 将排序后的嵌入向量和相应的余弦相似度打包成元组列表返回
    sorted_results = list(zip(sorted_embeddings.tolist(), sorted_similarities))

    return sorted_results


def plot_multiclass_precision_recall(
        y_score, y_true_untransformed, class_list, classifier_name
):
    """
    Precision-Recall plotting for a multiclass problem. It plots average precision-recall, per class precision recall and reference f1 contours.

    Code slightly modified, but heavily based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    n_classes = len(class_list)
    y_true = pd.concat(
        [pd.Series(y_true_untransformed == class_list[i]) for i in range(n_classes)], axis=1).values

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true.ravel(), y_score.ravel()
    )
    average_precision_micro = average_precision_score(y_true, y_score, average="micro")
    print(
        str(classifier_name)
        + " - Average precision score over all classes: {0:0.2f}".format(
            average_precision_micro
        )
    )

    # setup plot details
    plt.figure(figsize=(9, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    l = None  # 初始化l，防止静态代码分析工具发出警告
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall_micro, precision_micro, color="gold", lw=2)
    lines.append(l)
    labels.append(
        "average Precision-recall (auprc = {0:0.2f})" "".format(average_precision_micro)
    )

    for i in range(n_classes):
        (l,) = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append(
            "Precision-recall for class `{0}` (auprc = {1:0.2f})"
            "".format(class_list[i], average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{classifier_name}: Precision-Recall curve for each class")
    plt.legend(lines, labels)


def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)


def pca_components_from_embeddings(
        embeddings: List[List[float]], n_components=2
) -> np.ndarray:
    """Return the PCA components of a list of embeddings."""
    pca = PCA(n_components=n_components)
    array_of_embeddings = np.array(embeddings)
    return pca.fit_transform(array_of_embeddings)


def tsne_components_from_embeddings(
        embeddings: List[List[float]], n_components=2, **kwargs
) -> np.ndarray:
    """Returns t-SNE components of a list of embeddings."""
    # use better defaults if not specified
    if "init" not in kwargs.keys():
        kwargs["init"] = "pca"
    if "learning_rate" not in kwargs.keys():
        kwargs["learning_rate"] = "auto"
    tsne = TSNE(n_components=n_components, **kwargs)
    array_of_embeddings = np.array(embeddings)
    return tsne.fit_transform(array_of_embeddings)


def chart_from_components(
        components: np.ndarray,
        labels: Optional[List[str]] = None,
        strings: Optional[List[str]] = None,
        x_title="Component 0",
        y_title="Component 1",
        mark_size=5,
        **kwargs,
):
    """Return an interactive 2D chart of embedding components."""
    empty_list = ["" for _ in components]
    data = pd.DataFrame(
        {
            x_title: components[:, 0],
            y_title: components[:, 1],
            "label": labels if labels else empty_list,
            "string": ["<br>".join(tr.wrap(string, width=30)) for string in strings]
            if strings
            else empty_list,
        }
    )
    chart = px.scatter(
        data,
        x=x_title,
        y=y_title,
        color="label" if labels else None,
        symbol="label" if labels else None,
        hover_data=["string"] if strings else None,
        **kwargs,
    ).update_traces(marker=dict(size=mark_size))
    return chart


def chart_from_components_3D(
        components: np.ndarray,
        labels: Optional[List[str]] = None,
        strings: Optional[List[str]] = None,
        x_title: str = "Component 0",
        y_title: str = "Component 1",
        z_title: str = "Compontent 2",
        mark_size: int = 5,
        **kwargs,
):
    """Return an interactive 3D chart of embedding components."""
    empty_list = ["" for _ in components]
    data = pd.DataFrame(
        {
            x_title: components[:, 0],
            y_title: components[:, 1],
            z_title: components[:, 2],
            "label": labels if labels else empty_list,
            "string": ["<br>".join(tr.wrap(string, width=30)) for string in strings]
            if strings
            else empty_list,
        }
    )
    chart = px.scatter_3d(
        data,
        x=x_title,
        y=y_title,
        z=z_title,
        color="label" if labels else None,
        symbol="label" if labels else None,
        hover_data=["string"] if strings else None,
        **kwargs,
    ).update_traces(marker=dict(size=mark_size))
    return chart


embedding_cache_path = str(get_project_root() / "data/embeddings_cache.pkl")
# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}


# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def get_embedding_with_cache(string: str,
                             embedding_model="text-embedding-3-small",
                             ) -> List[float]:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, embedding_model) not in embedding_cache.keys():
        embedding_cache[(string, embedding_model)] = get_embedding_without_cache(string, embedding_model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, embedding_model)]


def get_most_similar_strings_by_index(source_strings: List[str],
                                      compare_string_index: int,
                                      k: int = 1,
                                      embedding_model="text-embedding-3-small",
                                      ) -> List[Tuple[str, float]]:
    return get_most_similar_strings(source_strings, source_strings[compare_string_index], k, embedding_model)


def get_most_similar_strings(source_strings: List[str],
                             compare_string: str,
                             k: int = 1,
                             embedding_model="text-embedding-3-small",
                             ) -> List[Tuple[str, float]]:
    """Return the k nearest neighbors of a given string, along with their cosine similarities."""
    # Get the embedding of the compare string
    compare_embedding = get_embedding_with_cache(compare_string, embedding_model=embedding_model)

    # Get embeddings for all source strings
    source_embeddings = [get_embedding_with_cache(source_string, embedding_model=embedding_model) for source_string in
                         source_strings]

    # Use sorted_by_similar_np to get sorted embeddings and their similarities
    sorted_results = sorted_by_similar_np(compare_embedding, source_embeddings, desc=True)

    # Print and collect the k most similar strings and their similarities
    similar_strings_with_similarity: List[Tuple[str, float]] = []
    k_counter = 0  # Initialize a counter for tracking how many similar strings have been processed

    for i, (sorted_embedding, similarity) in enumerate(sorted_results):
        # Find the original string corresponding to the sorted embedding
        original_index = source_embeddings.index(sorted_embedding)
        original_string = source_strings[original_index]

        # Skip the compare string itself
        if original_string == compare_string:
            continue

        # Limit to the top k similar strings
        if k_counter >= k:
            break

        # Print out the similar strings and their similarities
        print(f"""
        --- Recommendation #{k_counter + 1} (nearest neighbor {k_counter + 1} of {k}) ---
        String: {original_string}
        Similarity: {similarity:.3f}
        """)

        similar_strings_with_similarity.append((original_string, similarity))
        k_counter += 1

    return similar_strings_with_similarity


def predict_with_zero_shot(
        to_predict_texts: List[str],
        labels_for: List[str],
        embedding_model: str = "text-embedding-3-small",
        confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    使用零样本学习对文本进行分类。

    参数:
    - to_classify_texts: 待分类的文本列表。
    - classify_labels: 分类标签列表。
    - embedding_model: 使用的嵌入模型名称。
    - quality_threshold: 置信度阈值。

    返回:
    - 一个DataFrame，包含每个文本的预测标签和置信度。
    """

    # 获取标签的嵌入向量
    label_embeddings = [get_embedding_with_cache(label, embedding_model) for label in labels_for]

    predictions = []
    confidences = []

    for text in to_predict_texts:
        # 获取待分类文本的嵌入向量
        text_embedding = get_embedding_with_cache(text, embedding_model)

        # 计算文本嵌入向量与每个标签嵌入向量的相似度
        similarities = [
            np.dot(text_embedding, label_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(label_embedding))
            for label_embedding in label_embeddings]

        # 选择相似度最高的标签作为预测结果
        max_similarity_index = np.argmax(similarities)
        prediction = labels_for[max_similarity_index]
        confidence = similarities[max_similarity_index]

        # 只有在置信度超过阈值时才接受预测
        if confidence >= confidence_threshold:
            predictions.append(prediction)
            confidences.append(confidence)
        else:
            predictions.append("uncertain")
            confidences.append(confidence)

    # 创建包含预测结果的DataFrame
    results_df = pd.DataFrame({
        'text': to_predict_texts,
        'predicted_label': predictions,
        'confidence': confidences
    })

    return results_df


def predict_with_sample(
        samples_df: pd.DataFrame,
        samples_target_col: str,
        to_predict_texts: List[str],
        embedding_model_name: str = "text-embedding-3-small",
        quality_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    带有预测质量检查的通用预测方法

    参数:
    - samples_df: DataFrame，包含特征和目标列。
    - samples_target_col: 目标列的名称。
    - to_predict_texts: 待预测的文本列表。
    - embedding_model_name: 使用的嵌入模型名称。
    - prediction_model: 用于预测的模型实例。
    - quality_threshold: 预测质量的阈值。

    返回:
    - 包含预测结果的DataFrame，包含每个预测数据的置信度信息。

    抛出:
    - ValueError: 如果模型的预测质量未达到预设阈值。
    """

    # 生成训练和测试数据嵌入向量
    samples_df['embedding'] = samples_df['text_column'].apply(
        lambda x: get_embedding_with_cache(x, embedding_model_name))
    X = np.array(samples_df['embedding'].tolist())
    y = samples_df[samples_target_col].values

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化并训练模型
    prediction_model = RandomForestClassifier(random_state=42)
    prediction_model.fit(X_train, y_train)

    # 验证模型质量
    preds_test = prediction_model.predict(X_test)
    quality = f1_score(y_test, preds_test, average='macro')
    if quality < quality_threshold:
        raise ValueError(f"Model quality below threshold: {quality} < {quality_threshold}")

    # 对待预测文本生成嵌入向量并预测
    to_predict_embeddings = np.array(
        [get_embedding_with_cache(text, embedding_model_name) for text in to_predict_texts])
    probas = prediction_model.predict_proba(to_predict_embeddings)

    # 最可能的类别和相应的置信度
    preds = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)

    # 创建包含预测结果和置信度的DataFrame
    results_df = pd.DataFrame({
        'text': to_predict_texts,
        'predicted': [prediction_model.classes_[i] for i in preds],  # 将索引转换为类别标签
        'confidence': confidences
    })

    return results_df


def calculate_percentile_cosine_similarity(data: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    # 计算不同聚类之间整体关系的亲疏，注意：这种相似度比较只是关于文本语义的相似度比较，而不是因果关系的概率比较，
    # 比如在推荐系统中，我们可能想要了解用户的评论文本与产品描述文本之间的语义相似性，或者在客户服务中分析客户反馈
    # 与已知问题之间的相似度。然而，对于关心不同类别数据之间如何随时间变化而产生互动的情况，我们需要另一种方法，
    # 另外，目前算法会受不平衡的个体关系的影响较大，必要时要对个体的权重影响进行考量。
    embeddings = {
        category: data[category].apply(get_embedding_with_cache).tolist()
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