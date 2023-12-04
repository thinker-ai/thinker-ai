import base64

from thinker_ai.agent.agent import Assistant, DataModel, Agent
from thinker_ai.utils.Speaker import Speaker

template = """
{history}
人类: {human_input}
AI助理:"""


class SpeakAgent(Agent):
    def __init__(self, user_id: str, assistant: Assistant):
        super().__init__(user_id, assistant)
        self.speaker = Speaker()

    def ask_for_text_audio(self, msg: str) -> DataModel:
        text = super().ask_for_text(msg)
        ai_audio = self.speaker.take_speak_audio(text[0])
        audio_base64 = base64.b64encode(ai_audio.getvalue()).decode()  # Encode ai_audio to Base64
        return DataModel(text=text, audio=audio_base64)
