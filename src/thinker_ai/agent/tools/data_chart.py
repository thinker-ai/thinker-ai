from typing import Optional, List
import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import textwrap as tr


def plot_precision_recall(precision, recall, average_precision, precision_micro, recall_micro, average_precision_micro,
                          class_list, classifier_name, f_scores=None, figsize=(9, 10), line_width=2,
                          color_map=None, alpha=0.2, xlim=(0.0, 1.0), ylim=(0.0, 1.05)):
    """
    绘制每个类别的 precision-recall 曲线，以及微平均的 precision-recall 曲线。

    :param precision: 各个类别的精度
    :param recall: 各个类别的召回率
    :param average_precision: 各个类别的平均精度
    :param precision_micro: 微平均精度
    :param recall_micro: 微平均召回率
    :param average_precision_micro: 微平均的平均精度
    :param class_list: 类别列表
    :param classifier_name: 分类器名称
    :param f_scores: 等值 F1 分数的范围 (默认为 [0.2, 0.8])
    :param figsize: 图像的大小 (默认是 9x10)
    :param line_width: 线条宽度 (默认是 2)
    :param color_map: 颜色映射，用于不同类别的曲线 (默认为 None)
    :param alpha: 等值 F1 曲线的透明度 (默认是 0.2)
    :param xlim: x 轴的限制 (默认为 [0.0, 1.0])
    :param ylim: y 轴的限制 (默认为 [0.0, 1.05])
    """
    # 设置绘图
    plt.figure(figsize=figsize)

    # 如果没有提供 f_scores，使用默认范围
    if f_scores is None:
        f_scores = np.linspace(0.2, 0.8, num=4)

    lines = []
    labels = []

    # 绘制 iso-f1 curves
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=alpha)
        plt.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))
        lines.append(l)
    labels.append("iso-f1 curves")

    # 绘制 micro-average 曲线
    l, = plt.plot(recall_micro, precision_micro, color="gold", lw=line_width)
    lines.append(l)
    labels.append(f"average Precision-recall (auprc = {average_precision_micro:0.2f})")

    # 绘制每个类别的 precision-recall 曲线
    for i in range(len(class_list)):
        color = color_map[i] if color_map else None  # 使用 color_map 如果提供了
        l, = plt.plot(recall[i], precision[i], lw=line_width, color=color)
        lines.append(l)
        labels.append(f"Precision-recall for class `{class_list[i]}` (auprc = {average_precision[i]:0.2f})")

    # 设置图例和图表样式
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{classifier_name}: Precision-Recall curve for each class")
    plt.legend(lines, labels)
    plt.show()


def chart_from_components_2D(
        components: np.ndarray,
        labels: Optional[List[str]] = None,
        strings: Optional[List[str]] = None,
        x_title="x_title",
        y_title="y_title",
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
        x_title: str = "x_title",
        y_title: str = "y_title",
        z_title: str = "z_title",
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
