import json
import re
from typing import Dict, List, Optional, ClassVar

from pydantic import BaseModel, Field

from thinker_ai.agent.provider.llm_schema import Message
from thinker_ai.common.logs import logger
from thinker_ai.configs.config import config
from thinker_ai.configs.const import DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_SIZE


class BrainMemory(BaseModel):
    history: List[Message] = Field(default_factory=list)
    knowledge: List[Message] = Field(default_factory=list)
    historical_summary: str = ""
    last_history: Message = None
    is_dirty: bool = False
    last_talk: Optional[str] = None
    cacheable: bool = True
    cache_key: Optional[str] = None  # 用于存储缓存键

    cache: ClassVar[Dict[str, str]] = {}  # 内存缓存

    class Config:
        arbitrary_types_allowed = True

    def add_talk(self, msg: Message):
        """
        添加用户消息。
        """
        msg.role = "user"
        self.add_history(msg)
        self.is_dirty = True

    def add_answer(self, msg: Message):
        """添加 LLM 的回复"""
        msg.role = "assistant"
        self.add_history(msg)
        self.is_dirty = True

    def get_knowledge(self) -> str:
        texts = [m.content for m in self.knowledge]
        return "\n".join(texts)

    @staticmethod
    async def loads(cache_key: str) -> "BrainMemory":
        if not cache_key:
            return BrainMemory(cache_key=cache_key)
        v = BrainMemory.cache.get(cache_key)
        logger.debug(f"Cache GET {cache_key} {v}")
        if v:
            bm = BrainMemory.parse_raw(v)
            bm.is_dirty = False
            bm.cache_key = cache_key
            return bm
        return BrainMemory(cache_key=cache_key)

    async def dumps(self):
        if not self.is_dirty:
            return
        if not self.cache_key:
            return False
        v = self.model_dump_json()
        if self.cacheable:
            BrainMemory.cache[self.cache_key] = v
            logger.debug(f"Cache SET {self.cache_key} {v}")
        self.is_dirty = False

    @staticmethod
    def to_cache_key(prefix: str, user_id: str, chat_id: str):
        return f"{prefix}:{user_id}:{chat_id}"

    async def set_history_summary(self, history_summary):
        if self.historical_summary == history_summary:
            if self.is_dirty:
                await self.dumps()
                self.is_dirty = False
            return

        self.historical_summary = history_summary
        self.history = []
        await self.dumps()
        self.is_dirty = False

    def add_history(self, msg: Message):
        self.history.append(msg)
        self.last_history = msg
        self.is_dirty = True

    def exists(self, text) -> bool:
        for m in reversed(self.history):
            if m.content == text:
                return True
        return False

    @staticmethod
    def to_int(v, default_value):
        try:
            return int(v)
        except:
            return default_value

    def pop_last_talk(self):
        v = self.last_talk
        self.last_talk = None
        return v

    async def summarize(self, llm, max_words=200, keep_language: bool = False, limit: int = -1, **kwargs):
        texts = [self.historical_summary] if self.historical_summary else []
        texts.extend(m.content for m in self.history)
        text = "\n".join(texts)

        text_length = len(text)
        if limit > 0 and text_length < limit:
            return text
        summary = await self._summarize(text=text, llm=llm, max_words=max_words, keep_language=keep_language, limit=limit)
        if summary:
            await self.set_history_summary(history_summary=summary)
            return summary
        raise ValueError(f"text too long:{text_length}")

    async def get_title(self, llm, max_words=5, **kwargs) -> str:
        """生成文本标题"""
        summary = await self.summarize(llm=llm, max_words=max_words)
        language = config.language
        command = f"Translate the above summary into a {language} title of less than {max_words} words."
        summaries = [summary, command]
        msg = "\n".join(summaries)
        logger.debug(f"title ask:{msg}")
        response = await llm.aask(msg=msg, system_msgs=[], stream=False)
        logger.debug(f"title rsp: {response}")
        return response

    async def is_related(self, text1, text2, llm):
        context = f"## Paragraph 1\n{text2}\n---\n## Paragraph 2\n{text1}\n"
        rsp = await llm.aask(
            msg=context,
            system_msgs=[
                "You are a tool capable of determining whether two paragraphs are semantically related.",
                'Return "TRUE" if "Paragraph 1" is semantically relevant to "Paragraph 2", otherwise return "FALSE".'
            ],
            stream=False,
        )
        result = True if "TRUE" in rsp else False
        p2 = text2.replace("\n", "")
        p1 = text1.replace("\n", "")
        logger.info(f"IS_RELATED:\nParagraph 1: {p2}\nParagraph 2: {p1}\nRESULT: {result}\n")
        return result

    async def rewrite(self, sentence: str, context: str, llm):
        prompt = f"## Context\n{context}\n---\n## Sentence\n{sentence}\n"
        rsp = await llm.aask(
            msg=prompt,
            system_msgs=[
                'You are a tool augmenting the "Sentence" with information from the "Context".',
                "Do not supplement the context with information that is not present, especially regarding the subject and object.",
                "Return the augmented sentence.",
            ],
            stream=False,
        )
        logger.info(f"REWRITE:\nCommand: {prompt}\nRESULT: {rsp}\n")
        return rsp

    @staticmethod
    def extract_info(input_string, pattern=r"\[([A-Z]+)\]:\s*(.+)"):
        match = re.match(pattern, input_string)
        if match:
            return match.group(1), match.group(2)
        else:
            return None, input_string

    @property
    def is_history_available(self):
        return bool(self.history or self.historical_summary)

    @property
    def history_text(self):
        if len(self.history) == 0 and not self.historical_summary:
            return ""
        texts = [self.historical_summary] if self.historical_summary else []
        for m in self.history:
            if isinstance(m, Dict):
                t = Message(**m).content
            elif isinstance(m, Message):
                t = m.content
            else:
                continue
            texts.append(t)

        return "\n".join(texts)

    async def _summarize(self, text: str, llm, max_words=200, keep_language: bool = False, limit: int = -1) -> str:
        max_token_count = DEFAULT_MAX_TOKENS
        max_count = 100
        text_length = len(text)
        if limit > 0 and text_length < limit:
            return text
        summary = ""
        while max_count > 0:
            if text_length < max_token_count:
                summary = await self._get_summary(text=text, llm=llm, max_words=max_words, keep_language=keep_language)
                break

            padding_size = 20 if max_token_count > 20 else 0
            text_windows = self.split_texts(text, window_size=max_token_count - padding_size)
            part_max_words = min(int(max_words / len(text_windows)) + 1, 100)
            summaries = []
            for ws in text_windows:
                response = await self._get_summary(text=ws, llm=llm, max_words=part_max_words, keep_language=keep_language)
                summaries.append(response)
            if len(summaries) == 1:
                summary = summaries[0]
                break

            # 合并并重试
            text = "\n".join(summaries)
            text_length = len(text)

            max_count -= 1  # 安全措施，防止死循环
        return summary

    async def _get_summary(self, text: str, llm, max_words=20, keep_language: bool = False):
        """生成文本摘要"""
        if len(text) < max_words:
            return text
        system_msgs = [
            "You are a tool for summarizing and abstracting text.",
            f"Return the summarized text to less than {max_words} words.",
        ]
        if keep_language:
            system_msgs.append("The generated summary should be in the same language as the original text.")
        response = await llm.aask(msg=text, system_msgs=system_msgs, stream=False)
        logger.debug(f"{text}\nsummary rsp: {response}")
        return response

    @staticmethod
    def split_texts(text: str, window_size) -> List[str]:
        """将长文本拆分为滑动窗口的文本"""
        if window_size <= 0:
            window_size = DEFAULT_TOKEN_SIZE
        total_len = len(text)
        if total_len <= window_size:
            return [text]

        padding_size = 20 if window_size > 20 else 0
        windows = []
        idx = 0
        data_len = window_size - padding_size
        while idx < total_len:
            if window_size + idx > total_len:
                windows.append(text[idx:])
                break
            w = text[idx: idx + window_size]
            windows.append(w)
            idx += data_len

        return windows

    @classmethod
    def save_cache_to_disk(cls, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cls.cache, f)

    @classmethod
    def load_cache_from_disk(cls, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                cls.cache = json.load(f)
        except FileNotFoundError:
            cls.cache = {}