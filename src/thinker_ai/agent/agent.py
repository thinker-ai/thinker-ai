import time
from pprint import pprint
from typing import List, Optional, Any, Dict

from openai import OpenAI
from openai.pagination import SyncCursorPage
from openai.types.beta import AssistantToolParam, Thread
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import Message
from pydantic import BaseModel

from thinker_ai.llm import gpt


class DataModel(BaseModel):
    class Config:
        extra = "allow"  # 允许额外的字段

    def __iter__(self):
        return iter(self.dict().values())


class Agent:
    user_id: str
    threads:Dict[str,Thread]={}
    assistant: Assistant
    client: OpenAI = gpt.llm

    def __init__(self, user_id: str, assistant: Assistant):
        self.user_id = user_id
        self.assistant = assistant

    def create_thread(self,topic:str)->Thread:
         thread=self.client.beta.threads.create()
         self.threads[topic]=thread
         return thread

    @property
    def name(self):
        return self.assistant.name

    @property
    def id(self):
        return self.assistant.id

    def add_files(self, file_ids: List[str]):
        self.assistant.file_ids.extend(file_ids)

    def add_file(self, file_id: str):
        self.assistant.file_ids.append(file_id)

    def remove_file(self, file_id: str):
        self.assistant.file_ids.remove(file_id)

    def add_tools(self, tools: List[AssistantToolParam]):
        self.assistant.tools.extend(tools)

    def add_tool(self, tool: AssistantToolParam):
        self.assistant.tools.append(tool)

    def remove_tool(self, tool: AssistantToolParam):
        self.assistant.tools.remove(tool)

    def _create_message(self,topic:str, content) -> Message:
        thread=self.threads.get(topic)
        if thread is None:
            thread=self.create_thread(topic)
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
            file_ids=self.assistant.file_ids
        )
        return message

    def _ask_for_messages(self, topic:str,content: str) -> SyncCursorPage[Message]:
        message: Message = self._create_message(topic,content)
        run = self.client.beta.threads.runs.create(
            thread_id=message.thread_id,  # 指定运行的会话
            assistant_id=self.id  # 指定运行的实例
        )
        print("checking assistant status. ")
        while True:
            run = self.client.beta.threads.runs.retrieve(thread_id=message.thread_id, run_id=run.id)
            if run.status == "completed":
                pprint("done!")
                messages = self.client.beta.threads.messages.list(thread_id=message.thread_id)
                break
            else:
                pprint("in progress...")
                time.sleep(5)
        return messages

    def ask(self,topic:str, content: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        messages = self._ask_for_messages(topic,content)
        for message in messages:
            if message.role == "user":
                continue
            if message.content[0].type == "text":
                self._do_with_text(message, "text", result)
            if message.content[0].type == "image_file":
                self._do_with_image(message, "image_file", result)
        return result

    def _do_with_text(self, message, result_type: str, result: Dict[str, Any]):
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):
            # Replace the text with a footnote
            message_content.value = message_content.value.replace(annotation.text, f' [{index}]')
            # Gather citations based on annotation attributes
            file_citation = getattr(annotation, 'file_citation', None)
            if file_citation:
                cited_file = self.client.files.retrieve(file_citation.file_id)
                citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
            else:
                file_path = getattr(annotation, 'file_path', None)
                if file_path:
                    cited_file = self.client.files.retrieve(file_path.file_id)
                    citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
        # Add footnotes to the end of the message before displaying to user
        message_content.value += '\n' + '\n'.join(citations)
        result[result_type] = message_content.value

    def _do_with_image(self, message, result_type: str, result: Dict[str, Any]):
        image_file = message.content[0].image_file
        if image_file:
            image_data = self.client.files.content(image_file.file_id)
            image_data_bytes = image_data.read()
            result[result_type] = image_data_bytes
