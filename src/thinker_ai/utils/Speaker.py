import io
import re
import edge_tts


class Speaker:
    def clean_text(self,text):
        # 去除HTML标签
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        # 去除换行符和制表符
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        return text

    async def take_speak_audio(self, text: str) -> io.BytesIO:
        text = self.clean_text(text)
        communicate = edge_tts.Communicate(text=text, voice="zh-CN-XiaoxiaoNeural")
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        audio_stream.seek(0)
        return audio_stream
