from typing import Iterable, Dict, List


class Memory:

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        # 将storage定义为一个字典，键为主题（topic），值为属于该主题的消息列表
        self.storage: Dict[str, List[str]] = {}

    def add(self, topic: str, message: str):
        # 如果主题在storage中不存在，则初始化一个空列表
        if topic not in self.storage:
            self.storage[topic] = []
        # 将消息添加到对应主题的列表中
        self.storage[topic].append(message)

    def add_batch(self, topic: str, messages: Iterable[str]):
        # 对批量消息进行迭代，逐一添加
        for message in messages:
            self.add(topic, message)

    def get_by_keyword(self, topic: str, keyword: str) -> List[str]:
        # 仅在指定主题中搜索包含关键字的消息
        if topic in self.storage:
            return [message for message in self.storage[topic] if keyword in message]
        else:
            return []

    def delete(self, topic: str, message: str):
        # 如果消息存在于指定主题中，则删除该消息
        if topic in self.storage and message in self.storage[topic]:
            self.storage[topic].remove(message)
            # 如果删除消息后主题下没有任何消息，可以选择删除该主题键
            if not self.storage[topic]:
                self.del_topic(topic)

    def del_topic(self, topic: str):
        # 清空主题存储
        if topic in self.storage:
            del self.storage[topic]

    def clear(self):
        # 清空所有存储
        self.storage = {}
