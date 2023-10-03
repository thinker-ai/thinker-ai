import base64
from io import BytesIO
from typing import Dict, Tuple

from langchain import LLMChain, PromptTemplate, OpenAI
from pydantic import BaseModel
from langchain.memory import ConversationBufferWindowMemory

from thinker_ai.actions.action import BaseAction, Criteria
from thinker_ai.agent.agent import Agent
from thinker_ai.utils.Speaker import Speaker

template = """
{history}
人类: {human_input}
AI助理:"""


class SpeakAgent(Agent):
    def __init__(self, actions: Dict[str, BaseAction], criteria: Criteria):
        super().__init__(actions, criteria)
        self.speaker = Speaker()

    async def ask(self, msg: str) -> tuple[str, str]:
        text = self._ai_process(msg)
        ai_audio = await self.speaker.take_speak_audio(text)
        audio_base64 = base64.b64encode(ai_audio.getvalue()).decode()  # Encode ai_audio to Base64
        return text, audio_base64

    # 创建一个全局的消息队列，用于存储所有的输入和输出
    speaker: Speaker
    instance_id: str
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=30),
    )


    def _ai_process(self, msg: str) -> str:
        response_text = self.chatgpt_chain.predict(human_input=msg)
        return response_text
