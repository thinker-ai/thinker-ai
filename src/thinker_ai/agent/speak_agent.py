import base64

from thinker_ai.agent.agent import Agent, DataModel
from thinker_ai.utils.Speaker import Speaker

template = """
{history}
人类: {human_input}
AI助理:"""


class SpeakAgent(Agent):
    def __init__(self,name:str):
        super().__init__(name)
        self.speaker = Speaker()

    async def ask(self, msg: str) -> DataModel:
        [text] = await super().ask(msg)
        ai_audio = await self.speaker.take_speak_audio(text)
        audio_base64 = base64.b64encode(ai_audio.getvalue()).decode()  # Encode ai_audio to Base64
        return DataModel(text=text, audio=audio_base64)

