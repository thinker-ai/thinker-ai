from __future__ import annotations

import base64
import json
import time
from typing import List, Any, Dict, Callable, Optional, Type, Union, Literal, Iterable

from langchain_core.tools import BaseTool
from openai.types.beta import Thread, CodeInterpreterTool, FileSearchTool, CodeInterpreterToolParam, FileSearchToolParam
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import Message, Text, Run
from pydantic import BaseModel

from thinker_ai.agent.functions.functions_register import FunctionsRegister
from thinker_ai.agent.llm import gpt
from thinker_ai.agent.llm.embeddings import get_most_similar_strings
from thinker_ai.agent.llm.function_call import FunctionCall
from thinker_ai.common.common import show_json


class AssistantAgent:
    user_id: str
    threads: Dict[str, Thread] = {}
    functions_register = FunctionsRegister()
    client = gpt.llm

    def __init__(self, assistant: Assistant):
        self.assistant = assistant

    @classmethod
    def from_instance(cls, assistant: Assistant):
        return cls(assistant)

    @classmethod
    def from_id(cls, assistant_id: str):
        assistant = gpt.llm.beta.assistants.retrieve(assistant_id)
        return cls(assistant)

    @property
    def name(self) -> str:
        return self.assistant.name

    @property
    def tools(self) -> List[Dict]:
        return self.assistant.tools

    @property
    def file_ids(self) -> List[str]:
        return self.assistant.file_ids

    def set_instructions(self, instructions: str):
        self.client.beta.assistants.update(self.assistant.id, instructions=instructions)

    def register_file_id(self, file_id: str):
        exist_file_ids: List[str] = self.assistant.tool_resources.code_interpreter.file_ids
        for exist_file_id in exist_file_ids:
            if exist_file_id == file_id:
                return
        exist_file_ids.append(file_id)
        self.client.beta.assistants.update(assistant_id=self.assistant.id,
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
        self.client.beta.assistants.update(assistant_id=self.assistant.id,
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
                self.client.beta.assistants.update(assistant_id=self.assistant.id, tool_resources={
                                               "code_interpreter": {
                                                   "file_ids": exist_file_ids
                                               }
                                           })

    def remove_vector_store_id(self, vector_store_id: str):
        exist_vector_store_ids: List[str] = self.assistant.tool_resources.file_search.vector_store_ids
        for exist_vector_store_id in exist_vector_store_ids:
            if exist_vector_store_id == vector_store_id:
                exist_vector_store_ids.remove(exist_vector_store_id)
                self.client.beta.assistants.update(assistant_id=self.assistant.id, tool_resources={
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
            if key == "text":
                markdown_line += f"{value}"
            elif key == "image_file":
                # 假设 value 是图像数据的字节流
                # 将字节流转换为 base64 编码的字符串
                image_base64 = base64.b64encode(value).decode('utf-8')
                # 使用 Markdown 的图片语法嵌入 base64 编码的图像
                markdown_line += f"![Image](data:image/png;base64,{image_base64})\n"
        markdown_lines.append(markdown_line)
        return "\n".join(markdown_lines)

    def _do_with_result(self, message: Message) -> Dict[str, Any]:
        result: Dict = {}
        if message.role == "user":
            raise Exception("未正常获取回复")
        for content in message.content:
            result: Dict = {}
            if content.type == "text":
                result["text"] = self._do_with_text_result(content.text)
            if content.type == "image_file":
                result["image_file"] = self._do_with_image_result(content.image_file)
        return result

    def _ask_for_messages(self, topic: str, content: str) -> Message:
        thread, run = self._create_thread_and_run(topic, content)
        self._execute_function(run, thread.id)
        message = self._get_response(thread)
        show_json(message)
        return message

    def _print_step_details(self, run, thread_id: str):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=thread_id, run_id=run.id, order="asc"
        )
        for step in run_steps.data:
            step_details = step.step_details
            print(json.dumps(show_json(step_details), indent=4))

    def _wait_on_run(self, run, thread_id: str):
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
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
            run = self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
            run = self._wait_on_run(run, thread_id)
        return run

    def _create_thread_and_run(self, topic: str, content: str) -> [Thread, Run]:
        thread = self._create_thread(topic)
        run = self._submit_message(thread, content)
        run = self._wait_on_run(run, thread.id)
        return thread, run

    def _create_thread(self, topic: str) -> Thread:
        thread = self.threads.get(topic)
        if thread is None:
            thread = self.client.beta.threads.create()
            self.threads[topic] = thread
        return thread

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
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content,
            attachments= attachments
        )
        return self.client.beta.threads.runs.create(
            model=self.assistant.model,
            thread_id=thread.id,
            assistant_id=self.assistant.id,
            instructions=self.assistant.instructions,
            tools=self.assistant.tools,
            timeout=self.client.timeout
        )

    def _get_response(self, thread: Thread) -> Message:
        messages = self.client.beta.threads.messages.list(thread_id=thread.id, order="asc")
        return messages.data[-1]

    def _do_with_text_result(self, message_content: Text) -> str:
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

    def _do_with_image_result(self, image_file) -> bytes:
        image_data = self.client.files.content(image_file.file_id)
        image_data_bytes = image_data.read()
        return image_data_bytes

    def register_langchain_tool(self, tool: BaseTool):
        self.functions_register.register_langchain_tool(tool)
        self._update_functions()

    def register_function(self, func: Callable, args_schema: Optional[Type[BaseModel]]):
        self.functions_register.register_function(func, args_schema)
        self._update_functions()

    def register_langchain_tool_name(self, tool_name: str):
        self.functions_register.register_langchain_tool_names([tool_name])
        self._update_functions()

    def _register_native_tool(self, native_tool: Union[CodeInterpreterTool, FileSearchTool]):
        tools = self.assistant.tools
        for tool in tools:
            if tool.type == native_tool.type:
                return
        tools.append(native_tool)
        self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def _remove_native_tool(self, tool_type: Literal["code_interpreter", "file_search"]):
        tools = self.assistant.tools
        tools = [tool for tool in tools if tool.type != tool_type]
        self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def register_code_interpreter(self):
        self._register_native_tool(CodeInterpreterTool(type="code_interpreter"))

    def remove_code_interpreter(self):
        self._remove_native_tool("code_interpreter")

    def register_file_search(self):
        self._register_native_tool(FileSearchTool(type="file_search"))

    def remove_file_search(self):
        self._remove_native_tool("file_search")

    def _update_functions(self):
        tools = self.assistant.tools
        tools = [tool for tool in tools if tool.type != "function"]
        new_functions = self.functions_register.functions_schema
        for new_function in new_functions:
            new_tool = {
                "type": "function",
                "function": new_function
            }
            tools.append(new_tool)  # 添加到工具列表中
        # 使用更新后的工具列表更新助手
        self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def remove_functions(self):
        tools = self.assistant.tools
        tools = [tool for tool in tools if tool.type != "function"]
        self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def get_most_similar_strings(self,
                                 source_strings: List[str],
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
        source_strings = gpt.llm.files.retrieve(file_id)
        return get_most_similar_strings(source_strings, compare_string, k, embedding_model)