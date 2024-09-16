import networkx as nx
from typing import List

from thinker_ai.agent.memory.humanoid_memory.differentiable_neural_computer import DifferentiableNeuralComputer


class GraphDNC(DifferentiableNeuralComputer):
    """
    基于图结构的 DNC 实现，用于复杂关系的存储和推理。
    """

    def __init__(self):
        # 初始化图结构
        self.graph = nx.Graph()

    def store(self, inputs: List[str]):
        """
        将输入的信息作为节点存储到图中。
        假设每个输入项都是单独的信息，可以根据需要存储关系。
        """
        for item in inputs:
            self.graph.add_node(item)  # 将输入作为节点存储

    def retrieve(self, query: str) -> str:
        """
        从图结构中检索节点，如果存在则返回。
        """
        if query in self.graph.nodes:
            return query
        return "No relevant information found."

    def search(self, query: str) -> List[str]:
        """
        搜索与查询相关的节点。
        返回与该查询节点相邻的节点（邻居节点）。
        """
        if query in self.graph.nodes:
            neighbors = list(self.graph.neighbors(query))
            return neighbors
        return []

    def is_related(self, text1: str, text2: str) -> bool:
        """
        判断两个节点（文本）是否存在图中的直接关系（边）。
        """
        return self.graph.has_edge(text1, text2)

    def add_relation(self, text1: str, text2: str):
        """
        在图结构中添加两个文本之间的关系（边）。
        """
        self.graph.add_edge(text1, text2)

    def clear_memory(self):
        """
        清除图中的所有节点和边，重置图结构。
        """
        self.graph.clear()