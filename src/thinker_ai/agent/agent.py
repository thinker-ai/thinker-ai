import time
from pprint import pprint
from typing import List, Optional, Any, Dict

from openai import OpenAI
from openai.pagination import SyncCursorPage
from openai.types.beta.assistant import Tool, Assistant
from openai.types.beta.threads import ThreadMessage
from pydantic import BaseModel

from thinker_ai.llm import gpt
from thinker_ai.session.session_manager import SessionManager


class DataModel(BaseModel):
    class Config:
        extra = "allow"  # 允许额外的字段

    def __iter__(self):
        return iter(self.dict().values())


class Agent:
    user_id: str
    assistant: Assistant
    session_manager = SessionManager()
    client: OpenAI = gpt.llm

    def __init__(self, user_id: str, assistant: Assistant):
        self.user_id = user_id
        self.assistant = assistant

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

    def add_tools(self, tools: List[Tool]):
        self.assistant.tools.extend(tools)

    def add_tool(self, tool: Tool):
        self.assistant.tools.append(tool)

    def remove_tool(self, tool: Tool):
        self.assistant.tools.remove(tool)

    def _create_message(self, content, file_ids: List[str] = None) -> ThreadMessage:
        message = self.client.beta.threads.messages.create(
            thread_id=self.session_manager.get_or_create_session(self.user_id).id,
            role="user",
            content=content,
            file_ids=file_ids if file_ids is not None else []
        )
        return message

    def _ask_for_messages(self, content: str, file_ids: List[str] = None) -> SyncCursorPage[ThreadMessage]:
        message: ThreadMessage = self._create_message(content, file_ids)
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

    def ask(self, content: str, file_ids: List[str] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        messages = self._ask_for_messages(content, file_ids)
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
