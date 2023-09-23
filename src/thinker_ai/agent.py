import base64
from langchain import LLMChain, PromptTemplate, OpenAI
from pydantic import BaseModel
from langchain.memory import ConversationBufferWindowMemory
from thinker_ai.utils.Speaker import Speaker

template = """
{history}
人类: {human_input}
AI助理:"""


class DataModel(BaseModel):
    text: str
    audio_base64: str = None


class Agent:
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

    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.speaker = Speaker()

    async def chat_to(self,human_data: DataModel) -> DataModel:
        ai_data = self._ai_process(human_data)
        ai_audio = await self.speaker.take_speak_audio(ai_data.text)
        ai_data.audio_base64 = base64.b64encode(ai_audio.getvalue()).decode()  # Encode ai_audio to Base64
        return ai_data

    def _ai_process(self, human_input: DataModel) -> DataModel:
        response_text = self.chatgpt_chain.predict(human_input=human_input.text)
        return DataModel(text=response_text)

    def to_json(self):
        raise NotImplementedError()
