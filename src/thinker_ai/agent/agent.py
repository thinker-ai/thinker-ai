from __future__ import annotations

import json
import time
from typing import List, Any, Dict, Callable, Optional, Type, Union, Iterable

from langchain_core.tools import BaseTool
from openai import OpenAI, AsyncOpenAI
from openai.pagination import SyncCursorPage
from openai.types.beta import AssistantToolParam, Thread, FunctionToolParam
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import Message, Text, TextDelta, Run
from pydantic import BaseModel

from thinker_ai.agent.functions.functions_register import FunctionsRegister
from thinker_ai.agent.llm import gpt
from thinker_ai.agent.llm.function_call import FunctionCall
from thinker_ai.utils.common import show_json
from thinker_ai.utils.serializable import Serializable


class DataModel(BaseModel):
    class Config:
        extra = "allow"  # 允许额外的字段

    def __iter__(self):
        return iter(self.dict().values())


class Agent:
    id: str
    user_id: str
    threads: Dict[str, Thread] = {}
    assistant: Assistant
    functions_register = FunctionsRegister()

    def __init__(self, id: str, user_id: str, assistant: Assistant,threads: Dict[str, Thread], client: OpenAI):
        self.id = id
        self.client = client
        self.user_id = user_id
        self.assistant = assistant
        self.threads=threads

    @property
    def name(self):
        return self.assistant.name

    def update_files(self, file_ids: List[str]):
        self.assistant = self.client.beta.assistants.update(self.assistant.id, file_ids=file_ids)

    def ask(self, topic: str, content: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        messages: SyncCursorPage[Message] = self._ask_for_messages(topic, content)
        for message in messages:
            if message.role == "user":
                continue
            if message.content[0].type == "text" or message.content[0].type == "function":
                result["text"] = self._do_with_text(message.content[0].text)
            if message.content[0].type == "image_file":
                result["image_file"] = self._do_with_image(message.content[0].image_file)
        return result

    def _ask_for_messages(self, topic: str, content: str) -> SyncCursorPage[Message]:
        thread, run = self._create_thread_and_run(topic, content)
        show_json(run)
        run = self._wait_on_run(run, thread.id)
        show_json(run)
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=thread.id, run_id=run.id, order="asc"
        )
        for step in run_steps.data:
            step_details = step.step_details
            print(json.dumps(show_json(step_details), indent=4))
        messages = self._get_response(thread)
        show_json(messages)
        return messages

    def _wait_on_run(self, run, thread_id: str):
        while run.status == "queued" or run.status == "in_progress" or "requires_action":
            if run.status == "requires_action":
                tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
                function_call = FunctionCall(name=tool_call.function.name,
                                             arguments=json.loads(tool_call.function.arguments))
                print("Function Name:", function_call.name)
                print("Function Arguments:", function_call.arguments)
                function = self.functions_register.get_function(function_call.name)
                result = function.invoke(function_call.arguments)
                run = self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=[
                        {
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(result),
                        }
                    ],
                )
                return run
            else:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id,
                )
                time.sleep(0.5)
        return run

    def _create_thread_and_run(self, topic: str, content: str) -> [Thread, Run]:
        thread = self._create_thread(topic)
        run = self._submit_message(thread, content)
        return thread, run

    def _create_thread(self, topic: str) -> Thread:
        thread = self.threads.get(topic)
        if thread is None:
            thread = self.client.beta.threads.create()
            self.threads[topic] = thread
        return thread

    def _submit_message(self, thread: Thread, content: str) -> Run:
        """在thread中创建message，然后创建Run，它负责向assistant提交thread"""
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
            file_ids=self.assistant.file_ids
        )
        return self.client.beta.threads.runs.create(
            model=self.assistant.model,
            thread_id=thread.id,
            assistant_id=self.assistant.id,
            instructions=self.assistant.instructions,
            tools=self.assistant.tools,
            timeout=self.client.timeout
        )

    def _get_response(self, thread: Thread):
        return self.client.beta.threads.messages.list(thread_id=thread.id, order="asc")

    def _do_with_text(self, message_content: Text) -> str:
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
        return message_content.value

    def _do_with_image(self, image_file) -> bytes:
        image_data = self.client.files.content(image_file.file_id)
        image_data_bytes = image_data.read()
        return image_data_bytes

    def register_langchain_tool(self, tool: BaseTool):
        self.functions_register.register_langchain_tool(tool)
        self._update_function()

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        self.functions_register.register_function(func, args_schema)
        self._update_function()

    def register_langchain_tool_name(self, tool_name: str):
        self.functions_register.register_langchain_tool_names([tool_name])
        self._update_function()

    def _update_function(self):
        tools = self.assistant.tools
        function_exists = False  # 用于跟踪是否已经有一个function类型的工具

        # 更新已有的function工具
        for tool in tools:
            if tool.type == "function":
                tool.function = self.functions_register.functions_schema[0]
                function_exists = True  # 发现已有的function工具，更新标记

        # 如果没有发现function类型的工具，添加一个新的
        if not function_exists:
            new_function_tool = {
                "type": "function",
                "function": self.functions_register.functions_schema[0]
            }
            tools.append(new_function_tool)  # 添加到工具列表中

        # 使用更新后的工具列表更新助手
        self.assistant = self.client.beta.assistants.update(self.assistant.id, tools=tools)
