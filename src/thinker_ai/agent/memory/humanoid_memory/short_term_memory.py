from typing import List

from thinker_ai.agent.memory.humanoid_memory.memory_network import MemoryNetwork
from thinker_ai.agent.provider.llm_schema import Message


class ShortTermMemory:
    """
    短期记忆模块，用于存储和管理近期的对话信息。
    """

    def __init__(self, memory_network: MemoryNetwork):
        self.messages: List[Message] = []
        self.memory_network = memory_network  # 初始化记忆网络

    def save(self):
        """
        保存短期记忆。
        """
        data = [msg.to_dict() for msg in self.messages]
        self.memory_network.save(data)

    def load(self):
        """
        加载短期记忆。
        """
        data = self.memory_network.load()
        if data:
            self.messages = [Message(**msg) for msg in data]
            # 将消息内容加载到记忆网络
            for msg in self.messages:
                self.memory_network.add_memory([msg.content])
        else:
            self.messages = []

    def add_message(self, msg: Message):
        self.messages.append(msg)
        # 将消息内容添加到记忆网络
        self.memory_network.add_memory([msg.content])

    def get_history(self) -> List[Message]:
        return self.messages

    def get_history_text(self) -> str:
        return "\n".join([f"{msg.role}: {msg.content}" for msg in self.messages])

    def get_recent_context(self, window_size: int = 5) -> List[Message]:
        return self.messages[-window_size:]

    def exists(self, text: str) -> bool:
        return any(text == msg.content for msg in self.messages)

    def is_related(self, text1: str, text2: str) -> bool:
        # 使用记忆网络判断相关性
        return self.memory_network.is_related(text1, text2)

    def rewrite(self, sentence: str, context: str) -> str:
        # 使用记忆网络进行重写（简化处理）
        return f"{context} {sentence}"

    def query_memory(self, question: str) -> str:
        """
        在记忆网络中查询信息。
        """
        return self.memory_network.query(question)

    def clear(self):
        self.messages.clear()
        self.memory_network.clear_memory()