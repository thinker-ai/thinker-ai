from typing import List

from thinker_ai.agent.memory.humanoid_memory.long_term_memory import LongTermMemory
from thinker_ai.agent.memory.humanoid_memory.persistence import MemoryPersistence
from thinker_ai.agent.memory.humanoid_memory.short_term_memory import ShortTermMemory
from thinker_ai.agent.provider.llm_schema import Message
from typing import List, Optional


class HumanoidBrainMemory:

    """
    类人脑记忆体，管理短期记忆（STM）和长期记忆（LTM）。
    """

    def __init__(self, owner_id: str, stm_persistence: MemoryPersistence, ltm_persistence: MemoryPersistence):
        self.id = owner_id  # 所有者ID
        self.stm = ShortTermMemory(persistence=stm_persistence)
        self.ltm = LongTermMemory(persistence=ltm_persistence)
        self.is_dirty = False

    # ...（其他方法保持不变）

    def save(self):
        """
        保存当前的记忆状态。
        """
        if self.is_dirty:
            self.stm.save()
            self.ltm.save()
            print("Memory saved.")
            self.is_dirty = False

    def load(self):
        """
        加载记忆状态。
        """
        self.stm.load()
        self.ltm.load()
        print("Memory loaded.")

    def add_talk(self, msg: Message):
        """
        添加用户的消息到记忆体。
        """
        msg.role = "user"
        self.stm.add_message(msg)
        if self.should_update_ltm(msg):
            self.ltm.store_message(msg.content, msg.content)
        self.is_dirty = True

    def add_answer(self, msg: Message):
        """
        添加智能体的回复到记忆体。
        """
        msg.role = "assistant"
        self.stm.add_message(msg)
        if self.should_update_ltm(msg):
            self.ltm.store_message(msg.content, msg.content)
        self.is_dirty = True

    def get_history(self) -> List[Message]:
        """
        获取当前对话历史记录（短期记忆）。
        """
        return self.stm.get_history()

    def get_history_text(self) -> str:
        """
        获取对话历史的文本形式。
        """
        return self.stm.get_history_text()

    def summarize(self, max_words: int = 200) -> str:
        """
        对当前对话进行摘要，结合短期记忆和长期记忆。
        """
        recent_context = " ".join([msg.content for msg in self.stm.get_recent_context()])
        knowledge_list = self.ltm.search_knowledge(recent_context)
        knowledge_text = " ".join(knowledge_list)
        summary = f"Context: {recent_context}\nKnowledge: {knowledge_text}"
        return summary[:max_words]

    def get_title(self) -> str:
        """
        生成对话的标题，可能结合长期记忆中的知识。
        """
        if self.stm.get_history():
            first_msg = self.stm.get_history()[0]
            knowledge = self.ltm.retrieve_knowledge(first_msg.content)
            title = first_msg.content[:20]
            if knowledge:
                title += f" - {knowledge[:20]}"
            return title
        return "New Conversation"

    def is_related(self, text1: str, text2: str) -> bool:
        """
        判断两个文本是否相关，结合 STM 和 LTM。
        """
        # 使用 STM 判断
        if self.stm.is_related(text1, text2):
            return True
        # 使用 LTM 判断
        if self.ltm.is_related(text1, text2):
            return True
        return False

    def rewrite(self, sentence: str, context: str) -> str:
        """
        根据上下文和长期记忆重写句子。
        """
        # 使用 STM 进行重写
        rewritten_sentence = self.stm.rewrite(sentence, context)
        # 如果需要，可以进一步使用 LTM 进行优化
        rewritten_sentence = self.ltm.rewrite(rewritten_sentence, context)
        return rewritten_sentence

    def exists(self, text: str) -> bool:
        """
        检查给定的文本是否存在于历史记录中。
        """
        return self.stm.exists(text)

    def add_knowledge(self, key: str, value: str):
        """
        添加知识到长期记忆（LTM）。
        """
        self.ltm.store_message(key, value)
        self.is_dirty = True

    def get_knowledge(self, key: str) -> Optional[str]:
        """
        从长期记忆（LTM）中检索知识。
        """
        return self.ltm.retrieve_knowledge(key)

    def search_knowledge(self, query: str) -> List[str]:
        """
        搜索与查询相关的知识。
        """
        return self.ltm.search_knowledge(query)

    def clear_history(self):
        """
        清除对话历史记录（短期记忆）。
        """
        self.stm.clear()
        self.is_dirty = True

    def clear_knowledge(self):
        """
        清除长期记忆（LTM）中的知识。
        """
        self.ltm.clear()
        self.is_dirty = True

    def should_update_ltm(self, msg: Message) -> bool:
        """
        决定是否将消息存储到长期记忆。
        """
        return len(msg.content) > 50  # 例如，内容长度超过50的消息存储到LTM

    def get_last_talk(self) -> Optional[Message]:
        """
        获取最近的一条用户消息，不删除它。
        """
        if self.stm.messages:
            last_msg = self.stm.messages[-1]
            return last_msg
        return None

    def to_cache_key(self, prefix: str, user_id: str, chat_id: str) -> str:
        """
        生成用于缓存的键值。
        """
        return f"{prefix}:{user_id}:{chat_id}"

    @staticmethod
    def extract_info(input_string: str, pattern: str = r"\[([A-Z]+)\]:\s*(.+)") -> (Optional[str], str):
        """
        从输入字符串中提取信息。

        参数:
            input_string (str): 输入的字符串。
            pattern (str): 用于匹配的正则表达式。

        返回:
            Tuple[Optional[str], str]: 提取的标签和内容。
        """
        import re
        match = re.match(pattern, input_string)
        if match:
            return match.group(1), match.group(2)
        else:
            return None, input_string

    @property
    def is_history_available(self) -> bool:
        """
        检查是否有可用的历史记录。

        返回:
            bool: 如果有历史记录，返回 True，否则返回 False。
        """
        return bool(self.stm.get_history())

    @property
    def history_text(self) -> str:
        """
        获取历史记录的文本表示。

        返回:
            str: 历史记录的文本。
        """
        return self.stm.get_history_text()