from __future__ import annotations

import base64
import json
import time
from typing import List, Any, Dict, Callable, Optional, Type, Union, Literal

from langchain_core.tools import BaseTool
from openai.types.beta import Thread, CodeInterpreterTool, FileSearchTool, FunctionTool, AssistantTool
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import Message, Text, Run
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

from thinker_ai.agent.assistant import AssistantInterface
from thinker_ai.agent.openai_assistant import client
from thinker_ai.agent.tools.openai_functions_register import FunctionsRegister

from thinker_ai.agent.tools.embeddings import get_most_similar_strings
from thinker_ai.agent.tools.function_call import FunctionCall
from thinker_ai.common.common import show_json

from thinker_ai.context_mixin import ContextMixin


class OpenAiAssistant(AssistantInterface, ContextMixin):
    assistant: Assistant
    topic_threads: Dict[str, str] = {}
    functions_register: FunctionsRegister = FunctionsRegister()

    @property
    def name(self) -> str:
        return self.assistant.name

    @property
    def id(self) -> str:
        return self.assistant.id

    @property
    def tools(self) -> List[Dict]:
        return self.assistant.tools

    @property
    def file_ids(self) -> List[str]:
        return self.assistant.file_ids

    @classmethod
    def from_instance(cls, user_id: str, assistant: Assistant):
        return cls(user_id=user_id, assistant=assistant)

    @classmethod
    def from_id(cls, user_id: str, assistant_id: str):
        assistant = client.beta.assistants.retrieve(assistant_id)
        return cls(user_id=user_id, assistant=assistant)

    def set_instructions(self, instructions: str):
        client.beta.assistants.update(self.assistant.id, instructions=instructions)

    def register_file_id(self, file_id: str):
        exist_file_ids: List[str] = self.assistant.tool_resources.code_interpreter.file_ids
        for exist_file_id in exist_file_ids:
            if exist_file_id == file_id:
                return
        exist_file_ids.append(file_id)
        client.beta.assistants.update(assistant_id=self.assistant.id,
                                      tool_resources={
                                          "code_interpreter": {
                                              "file_ids": exist_file_ids
                                          }
                                      }
                                      )

    def register_vector_store_id(self, vector_store_id: str):
        exist_vector_store_ids: List[str] = self.assistant.tool_resources.file_search.vector_store_ids
        for exist_vector_store_id in exist_vector_store_ids:
            if exist_vector_store_id == vector_store_id:
                return
        exist_vector_store_ids.append(vector_store_id)
        client.beta.assistants.update(assistant_id=self.assistant.id,
                                      tool_resources={
                                          "file_search": {
                                              "vector_store_ids": exist_vector_store_ids
                                          }
                                      }
                                      )

    def remove_file_id(self, file_id: str):
        exist_file_ids: List[str] = self.assistant.tool_resources.code_interpreter.file_ids
        for exist_file_id in exist_file_ids:
            if exist_file_id == file_id:
                exist_file_ids.remove(exist_file_id)
                client.beta.assistants.update(assistant_id=self.assistant.id, tool_resources={
                    "code_interpreter": {
                        "file_ids": exist_file_ids
                    }
                })

    def remove_vector_store_id(self, vector_store_id: str):
        exist_vector_store_ids: List[str] = self.assistant.tool_resources.file_search.vector_store_ids
        for exist_vector_store_id in exist_vector_store_ids:
            if exist_vector_store_id == vector_store_id:
                exist_vector_store_ids.remove(exist_vector_store_id)
                client.beta.assistants.update(assistant_id=self.assistant.id, tool_resources={
                    "file_search": {
                        "vector_store_ids": exist_vector_store_ids
                    }
                })

    def ask(self, content: str, topic: str = "default") -> str:
        message: Message = self._ask_for_messages(topic, content)
        response_content = self._do_with_result(message)
        return self._to_markdown(response_content)

    def _to_markdown(self, content: Dict[str, Any]) -> str:
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

    def _ask_for_messages(self, topic: str, content: str) -> Message:
        topic_thread, run = self._get_or_create_thread_and_run(topic, content)
        self._execute_function(run, topic_thread.id)
        message = self._get_response(topic_thread)
        show_json(message)
        return message

    def _print_step_details(self, run, thread_id: str):
        run_steps = client.beta.threads.runs.steps.list(
            thread_id=thread_id, run_id=run.id, order="asc"
        )
        for step in run_steps.data:
            step_details = step.step_details
            print(json.dumps(show_json(step_details), indent=4))

    def _wait_on_run(self, run, thread_id: str):
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id,
            )
            time.sleep(0.5)
        self._print_step_details(run, thread_id)
        return run

    def _execute_function(self, run, thread_id):
        if run.status == "requires_action":
            tool_calls: List = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs: List[Dict] = []
            for tool_call in tool_calls:
                function_call = FunctionCall(name=tool_call.function.name,
                                             arguments=json.loads(tool_call.function.arguments))
                print("Function Name:", function_call.name)
                print("Function Arguments:", function_call.arguments)
                function = self.functions_register.get_function(function_call.name)
                if function is None:
                    raise Exception(f"function {function_call.name} not found")
                result = function.invoke(function_call.arguments)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(result),
                })
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
            run = self._wait_on_run(run, thread_id)
        return run

    def _get_or_create_thread_and_run(self, topic: str, content: str) -> [Thread, Run]:
        thread_id = self.topic_threads.get(topic)
        if thread_id:
            topic_thread = self.client.beta.threads.retrieve(thread_id)
        else:
            topic_thread = client.beta.threads.create()
            self.topic_threads[topic] = topic_thread.id
        run = self._submit_message(topic_thread, content)
        run = self._wait_on_run(run, topic_thread.id)
        return topic_thread, run

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
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
            attachments=attachments
        )
        return client.beta.threads.runs.create(
            model=self.assistant.model,
            thread_id=thread.id,
            assistant_id=self.assistant.id,
            instructions=self.assistant.instructions,
            tools=self.assistant.tools,
            timeout=client.timeout
        )

    def _get_response(self, thread: Thread) -> Message:
        messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
        return messages.data[-1]

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
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
            else:
                file_path = getattr(annotation, 'file_path', None)
                if file_path:
                    cited_file = client.files.retrieve(file_path.file_id)
                    citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
        # Add footnotes to the end of the message before displaying to user
        message_content.value += '\n' + '\n'.join(citations)
        return message_content.value

    def _do_with_image_result(self, image_file) -> bytes:
        image_data = client.files.content(image_file.file_id)
        image_data_bytes = image_data.read()
        return image_data_bytes

    def register_langchain_tool(self, tool: BaseTool):
        self.functions_register.register_langchain_tool(tool)
        self._add_functions()

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        self.functions_register.register_function(func, args_schema)
        self._add_functions()

    def register_langchain_tool_name(self, tool_name: str):
        self.functions_register.register_langchain_tool_names([tool_name])
        self._add_functions()

    def _register_native_tool(self, native_tool: Union[CodeInterpreterTool, FileSearchTool]):
        tools = self.assistant.tools
        for tool in tools:
            if tool.type == native_tool.type:
                return
        tools.append(native_tool)
        client.beta.assistants.update(self.assistant.id, tools=tools)

    def _remove_native_tool(self, tool_type: Literal["code_interpreter", "file_search"]):
        tools = self.assistant.tools
        tools = [tool for tool in tools if tool.type != tool_type]
        client.beta.assistants.update(self.assistant.id, tools=tools)

    def register_code_interpreter(self):
        self._register_native_tool(CodeInterpreterTool(type="code_interpreter"))

    def remove_code_interpreter(self):
        self._remove_native_tool("code_interpreter")

    def register_file_search(self):
        self._register_native_tool(FileSearchTool(type="file_search"))

    def remove_file_search(self):
        self._remove_native_tool("file_search")

    def _add_functions(self):
        tools = self.assistant.tools
        new_functions = self.functions_register.functions_schema
        update = False
        for new_function in new_functions:
            function_definition = FunctionDefinition(**new_function)
            function_tool = FunctionTool(function=function_definition, type="function")
            if not self.is_function_registered(function_tool.function.name):
                tools.append(function_tool)  # 添加到工具列表中
                update = True
        # 使用更新后的工具列表更新助手
        if update:
            client.beta.assistants.update(self.assistant.id, tools=tools)

    def remove_functions(self):
        tools = self.assistant.tools
        new_tools = []
        for tool in tools:
            if tool.type != "function":
                new_tools.append(tool)
        client.beta.assistants.update(self.assistant.id, tools=new_tools)

    def is_function_registered(self, name: str) -> bool:
        tools = self.assistant.tools
        for tool in tools:
            if tool.type == "function" and tool.function.name == name:
                return True
        return False

    @staticmethod
    def get_most_similar_strings(source_strings: List[str],
                                 compare_string: str,
                                 k: int = 1,
                                 embedding_model="text-embedding-3-small",
                                 ) -> list[tuple[str, float]]:
        return get_most_similar_strings(source_strings, compare_string, k, embedding_model)

    def get_most_similar_from_file(self,
                                   file_id: str,
                                   compare_string: str,
                                   k: int = 1,
                                   embedding_model="text-embedding-3-small",
                                   ) -> list[tuple[str, float]]:
        source_strings = client.files.retrieve(file_id)
        return get_most_similar_strings(source_strings, compare_string, k, embedding_model)
