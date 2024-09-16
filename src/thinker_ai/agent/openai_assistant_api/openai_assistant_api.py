from __future__ import annotations

import base64
import json
import time
from typing import List, Any, Dict, Union, Literal, cast

from httpx import HTTPStatusError
from openai import BadRequestError
from openai.types.beta import Thread, CodeInterpreterTool, FileSearchTool, FunctionTool
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import Message, Text, Run

from thinker_ai.agent.assistant_api import AssistantApi
from thinker_ai.agent.openai_assistant_api import openai

from thinker_ai.agent.tools.embedding_helper import EmbeddingHelper
from thinker_ai.agent.topic_repository.openai_topic_repository import OpenAiTopicInfoRepository
from thinker_ai.agent.topic_repository.topic_builder import OpenAiTopic
from thinker_ai.agent.topic_repository.topic_repository import TopicInfo
from thinker_ai.context_mixin import ContextMixin
from thinker_ai.common.common import show_json

embedding_helper = EmbeddingHelper()

class OpenAiAssistantApi(AssistantApi, ContextMixin):
    assistant: Assistant
    topic_info_repository: OpenAiTopicInfoRepository = OpenAiTopicInfoRepository.get_instance()

    @property
    def name(self) -> str:
        return self.assistant.name

    @name.setter
    def name(self, new_value):
        try:
            self.assistant = openai.client.beta.assistants.update(self.assistant.id, name=new_value)
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def add_topic(self, user_id: str, topic_name: str) -> Thread:
        try:
            topic_thread = openai.client.beta.threads.create()
            topic = OpenAiTopic(name=topic_name, assistant_id=self.id, thread_id=topic_thread.id)
            self.topic_info_repository.add(TopicInfo(user_id=user_id, topic=topic))
            return topic_thread
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def del_topic(self, user_id: str, topic_name):
        topic_info = self.topic_info_repository.get_by(user_id, topic_name)
        if topic_info:
            topic = cast(OpenAiTopic, topic_info.topic)
            try:
                openai.client.beta.threads.delete(topic.thread_id)
                self.topic_info_repository.delete(user_id, topic_name)
            except (HTTPStatusError, BadRequestError) as e:
                print(str(e))

    @property
    def id(self) -> str:
        return self.assistant.id

    @property
    def tools(self) -> list[CodeInterpreterTool | FileSearchTool | FunctionTool]:
        return self.assistant.tools

    @property
    def file_ids(self) -> List[str]:
        return self.assistant.file_ids

    @classmethod
    def from_instance(cls, assistant: Assistant):
        return cls(assistant=assistant)

    @classmethod
    def from_id(cls, assistant_id: str):
        try:
            assistant = openai.client.beta.assistants.retrieve(assistant_id)
            return cls(assistant=assistant)
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def set_instructions(self, instructions: str):
        try:
            openai.client.beta.assistants.update(self.assistant.id, instructions=instructions)
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def register_file_id(self, file_id: str):
        exist_file_ids: List[str] = self.assistant.tool_resources.code_interpreter.file_ids
        for exist_file_id in exist_file_ids:
            if exist_file_id == file_id:
                return
        try:
            exist_file_ids.append(file_id)
            openai.client.beta.assistants.update(assistant_id=self.assistant.id,
                                                 tool_resources={
                                                     "code_interpreter": {
                                                         "file_ids": exist_file_ids
                                                     }
                                                 }
                                                 )
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def register_vector_store_id(self, vector_store_id: str):
        exist_vector_store_ids: List[str] = self.assistant.tool_resources.file_search.vector_store_ids
        for exist_vector_store_id in exist_vector_store_ids:
            if exist_vector_store_id == vector_store_id:
                return
        try:
            exist_vector_store_ids.append(vector_store_id)
            openai.client.beta.assistants.update(assistant_id=self.assistant.id,
                                                 tool_resources={
                                                     "file_search": {
                                                         "vector_store_ids": exist_vector_store_ids
                                                     }
                                                 }
                                                 )
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def remove_file_id(self, file_id: str):
        exist_file_ids: List[str] = self.assistant.tool_resources.code_interpreter.file_ids
        for exist_file_id in exist_file_ids:
            if exist_file_id == file_id:
                try:
                    exist_file_ids.remove(exist_file_id)
                    openai.client.beta.assistants.update(assistant_id=self.assistant.id, tool_resources={
                        "code_interpreter": {
                            "file_ids": exist_file_ids
                        }
                    })
                except (HTTPStatusError, BadRequestError) as e:
                    print(str(e))

    def remove_vector_store_id(self, vector_store_id: str):
        exist_vector_store_ids: List[str] = self.assistant.tool_resources.file_search.vector_store_ids
        for exist_vector_store_id in exist_vector_store_ids:
            if exist_vector_store_id == vector_store_id:
                try:
                    exist_vector_store_ids.remove(exist_vector_store_id)
                    openai.client.beta.assistants.update(assistant_id=self.assistant.id, tool_resources={
                        "file_search": {
                            "vector_store_ids": exist_vector_store_ids
                        }
                    })
                except (HTTPStatusError, BadRequestError) as e:
                    print(str(e))

    def ask(self, user_id: str, content: str, topic_name: str = "default") -> [str, str]:
        message = self._ask_for_messages(user_id=user_id, topic_name=topic_name, content=content)
        response_content = self._do_with_result(message)
        return self._to_markdown(response_content)

    @staticmethod
    def _to_markdown(content: Dict[str, Any]) -> str:
        markdown_lines = []
        markdown_line = ""
        for key, value in content.items():
            if key.startswith("text"):
                markdown_line += f"{value}"
            elif key.startswith("image_file"):
                # 假设 value 是图像数据的字节流
                # 将字节流转换为 base64 编码的字符串
                image_base64 = base64.b64encode(value).decode('utf-8')
                # 使用 Markdown 的图片语法嵌入 base64 编码的图像
                markdown_line += f"![Image](data:image/png;base64,{image_base64})\n"
        markdown_lines.append(markdown_line)
        return "\n".join(markdown_lines)

    def _do_with_result(self, message: Message) -> Dict[str, Any]:
        if message.role == "user":
            raise Exception("未正常获取回复")
        result: Dict = {}
        text_index = 0
        image_index = 0
        for content in message.content:
            if content.type == "text":
                result[f"text_{text_index}"] = self._do_with_text_result(content.text)
                text_index += 1
            if content.type == "image_file":
                result[f"image_file_{image_index}"] = self._do_with_image_result(content.image_file)
                image_index += 1
        return result

    def _ask_for_messages(self, user_id: str, topic_name: str, content: str) -> Message:
        thread_id, run = self._get_or_create_thread_and_run(user_id=user_id, topic_name=topic_name, content=content)
        self._execute_function(run, thread_id)
        message = self._get_response(thread_id)
        show_json(message)
        return message

    @staticmethod
    def _print_step_details(run, thread_id: str):
        try:
            run_steps = openai.client.beta.threads.runs.steps.list(
                thread_id=thread_id, run_id=run.id, order="asc"
            )
            for step in run_steps.data:
                step_details = step.step_details
                print(json.dumps(show_json(step_details), indent=4))
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def _wait_on_run(self, run, thread_id: str) -> Run:
        try:
            while run.status == "queued" or run.status == "in_progress":
                run = openai.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id,
                )
                time.sleep(0.5)
            self._print_step_details(run, thread_id)
            return run
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def _execute_function(self, run, thread_id) -> Run:
        try:
            if run.status == "requires_action":
                tool_calls: List = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs: List[Dict] = []
                for tool_call in tool_calls:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    callable = openai.callables_register.get_langchain_tool(name)
                    if callable is None:
                        raise Exception(f"function {name} not found")
                    result = callable.invoke(arguments)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(result),
                    })
                run = openai.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
                run = self._wait_on_run(run, thread_id)
            return run
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def _get_or_create_thread_and_run(self, user_id: str, topic_name: str, content: str) -> tuple[str, Run]:
        try:
            topic_info = self.topic_info_repository.get_by(user_id=user_id, topic_name=topic_name)
            if topic_info:
                topic = cast(OpenAiTopic, topic_info.topic)
                topic_thread = openai.client.beta.threads.retrieve(topic.thread_id)
            else:
                topic_thread = self.add_topic(user_id, topic_name)
            run = self._submit_message(topic_thread, content)
            run = self._wait_on_run(run, topic_thread.id)
            return topic_thread.id, run
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def _submit_message(self, thread: Thread, content: str) -> Run:
        """在thread中创建message，然后创建Run，它负责向assistant提交thread"""
        attachments = []

        # Check if assistant.tool_resources exists and file_search is not None
        if self.assistant.tool_resources and self.assistant.tool_resources.code_interpreter:
            file_ids = self.assistant.tool_resources.code_interpreter.file_ids or []
            for file_id in file_ids:
                attachments.append({
                    "file_id": file_id,
                    "tools": [{"type": "code_interpreter"}]
                })
        try:
            openai.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=content,
                attachments=attachments
            )
            return openai.client.beta.threads.runs.create(
                model=self.assistant.model,
                thread_id=thread.id,
                assistant_id=self.assistant.id,
                instructions=self.assistant.instructions,
                tools=self.assistant.tools,
                timeout=openai.client.timeout
            )
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    @staticmethod
    def _get_response(thread_id: str) -> Message:
        try:
            messages = openai.client.beta.threads.messages.list(thread_id=thread_id, order="asc")
            return messages.data[-1]
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    @staticmethod
    def _do_with_text_result(message_content: Text) -> str:
        annotations = message_content.annotations
        citations = []
        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):
            # Replace the text with a footnote
            message_content.value = message_content.value.replace(annotation.text, f' [{index}]')
            # Gather citations based on annotation attributes
            file_citation = getattr(annotation, 'file_citation', None)
            if file_citation:
                cited_file = openai.client.files.retrieve(file_citation.file_id)
                citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
            else:
                file_path = getattr(annotation, 'file_path', None)
                if file_path:
                    cited_file = openai.client.files.retrieve(file_path.file_id)
                    citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
        # Add footnotes to the end of the message before displaying to user
        message_content.value += '\n' + '\n'.join(citations)
        return message_content.value

    @staticmethod
    def _do_with_image_result(image_file) -> bytes:
        image_data = openai.client.files.content(image_file.file_id)
        image_data_bytes = image_data.read()
        return image_data_bytes

    def _register_native_tool(self, native_tool: Union[CodeInterpreterTool, FileSearchTool]):
        tools = []
        for tool in self.assistant.tools:
            if tool.type == native_tool.type:
                return
            else:
                tools.append(tool)
        try:
            tools.append(native_tool)
            openai.client.beta.assistants.update(self.assistant.id, tools=tools)
            self.assistant.tools = tools
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def _remove_native_tool(self, tool_type: Literal["code_interpreter", "file_search"]):
        tools = [tool for tool in self.assistant.tools if tool.type != tool_type]
        try:
            openai.client.beta.assistants.update(self.assistant.id, tools=tools)
            self.assistant.tools = tools
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def register_code_interpreter(self):
        self._register_native_tool(CodeInterpreterTool(type="code_interpreter"))

    def remove_code_interpreter(self):
        self._remove_native_tool("code_interpreter")

    def register_file_search(self):
        self._register_native_tool(FileSearchTool(type="file_search"))

    def remove_file_search(self):
        self._remove_native_tool("file_search")

    def load_callables(self, callable_names: set[str]):
        tools = [tool for tool in self.assistant.tools]
        append = False
        for callable_name in callable_names:
            function_tool = openai.callables_register.get_function_tool(callable_name)
            if function_tool and not self.is_function_in_assistant(function_tool.function.name):
                tools.append(function_tool)
                append = True
        if append:
            try:
                openai.client.beta.assistants.update(assistant_id=self.assistant.id, tools=tools)
                self.assistant.tools = tools
            except (HTTPStatusError, BadRequestError) as e:
                print(str(e))

    def unload_callables(self, callable_names: set[str]):
        tool_names = {tool.function.name for tool in self.assistant.tools}
        retained_function_names = tool_names - callable_names
        retained_tools = []
        for tool in self.assistant.tools:
            for retained_name in retained_function_names:
                if tool.type != "function" or tool.function.name == retained_name:
                    retained_tools.append(tool)
        if len(retained_tools) < len(self.assistant.tools):
            try:
                openai.client.beta.assistants.update(self.assistant.id, tools=retained_tools)
                self.assistant.tools = retained_tools  # 用于AI调用的信息，此处为了提高性能，不再从远程获取
            except (HTTPStatusError, BadRequestError) as e:
                print(str(e))

    def unload_all_callables(self):
        tools = self.assistant.tools
        new_tools = []
        for tool in tools:
            if tool.type != "function":
                new_tools.append(tool)
        try:
            openai.client.beta.assistants.update(self.assistant.id, tools=new_tools)
            self.assistant.tools = new_tools
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))

    def is_function_in_assistant(self, name: str) -> bool:
        for tool in self.assistant.tools:
            if tool.type == "function" and tool.function.name == name:
                return True
        return False

    @staticmethod
    def get_most_similar_strings(source_strings: List[str],
                                 compare_string: str,
                                 k: int = 1,
                                 embedding_model="text-embedding-3-small",
                                 ) -> list[tuple[str, float]]:
        return embedding_helper.get_most_similar_strings(source_strings, compare_string, k, embedding_model)

    def get_most_similar_from_file(self,
                                   file_id: str,
                                   compare_string: str,
                                   k: int = 1,
                                   embedding_model="text-embedding-3-small",
                                   ) -> list[tuple[str, float]]:
        try:
            source_strings = openai.client.files.retrieve(file_id)
            return embedding_helper.get_most_similar_strings(source_strings, compare_string, k, embedding_model)
        except (HTTPStatusError, BadRequestError) as e:
            print(str(e))
