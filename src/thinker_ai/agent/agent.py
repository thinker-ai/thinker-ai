from __future__ import annotations

import json
import time
from typing import List, Any, Dict, Callable, Optional, Type, Union, Literal, Tuple

from langchain_core.tools import BaseTool
from openai import OpenAI
from openai.pagination import SyncCursorPage
from openai.types.beta import Thread, CodeInterpreterTool, RetrievalTool
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import Message, Text, Run
from pydantic import BaseModel

from thinker_ai.agent.functions.functions_register import FunctionsRegister
from thinker_ai.agent.llm import gpt
from thinker_ai.agent.llm.embeddings import get_most_similar_strings
from thinker_ai.agent.llm.function_call import FunctionCall
from thinker_ai.utils.common import show_json


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

    def __init__(self, id: str, user_id: str, assistant: Assistant, threads: Dict[str, Thread], client: OpenAI):
        self.id = id
        self.client = client
        self.user_id = user_id
        self.assistant = assistant
        self.threads = threads

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
        file_ids = self.assistant.file_ids
        for exist_file_id in file_ids:
            if exist_file_id == file_id:
                return
        file_ids.append(file_id)
        self.assistant = self.client.beta.assistants.update(self.assistant.id, file_ids=file_ids)

    def remove_file_id(self, file_id: str):
        file_ids = self.assistant.file_ids
        for exist_file_id in file_ids:
            if exist_file_id == file_id:
                file_ids.remove(exist_file_id)
                self.assistant = self.client.beta.assistants.update(self.assistant.id, file_ids=file_ids)

    def ask(self, topic: str, content: str) -> List[Dict[str, Any]]:
        messages: SyncCursorPage[Message] = self._ask_for_messages(topic, content)
        return self._do_with_result(messages)

    def _do_with_result(self, messages: SyncCursorPage[Message]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for message in messages.data:
            if message.role == "user":
                continue
            for content in message.content:
                result: Dict = {}
                if content.type == "text":
                    result["text"] = self._do_with_text_result(content.text)
                if content.type == "image_file":
                    result["image_file"] = self._do_with_image_result(content.image_file)
                results.append(result)
        return results

    def _ask_for_messages(self, topic: str, content: str) -> SyncCursorPage[Message]:
        thread, run = self._create_thread_and_run(topic, content)
        self._execute_function(run, thread.id)
        messages = self._get_response(thread)
        show_json(messages)
        return messages

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

    def _register_native_tool(self, native_tool: Union[CodeInterpreterTool, RetrievalTool]):
        tools = self.assistant.tools
        for tool in tools:
            if tool.type == native_tool.type:
                return
        tools.append(native_tool)
        self.assistant = self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def _remove_native_tool(self, tool_type: Literal["code_interpreter", "retrieval"]):
        tools = self.assistant.tools
        tools = [tool for tool in tools if tool.type != tool_type]
        self.assistant = self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def register_code_interpreter(self):
        self._register_native_tool(CodeInterpreterTool(type="code_interpreter"))

    def remove_code_interpreter(self):
        self._remove_native_tool("code_interpreter")

    def register_retrieval(self):
        self._register_native_tool(RetrievalTool(type="retrieval"))

    def remove_retrieval(self):
        self._remove_native_tool("retrieval")

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
        self.assistant = self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def remove_functions(self):
        tools = self.assistant.tools
        tools = [tool for tool in tools if tool.type != "function"]
        self.assistant = self.client.beta.assistants.update(self.assistant.id, tools=tools)

    def get_most_similar_strings(self,
                                 source_strings: List[str],
                                 compare_string: str,
                                 k: int = 1,
                                 embedding_model="text-embedding-3-small",
                                 ) -> list[tuple[str, float]]:
        return get_most_similar_strings(source_strings, compare_string, k, embedding_model)

    def get_most_similar_from_file(self,
                                 file_id:str,
                                 compare_string: str,
                                 k: int = 1,
                                 embedding_model="text-embedding-3-small",
                                 ) -> list[tuple[str, float]]:
        source_strings = gpt.llm.files.retrieve(file_id)
        return get_most_similar_strings(source_strings, compare_string, k, embedding_model)
