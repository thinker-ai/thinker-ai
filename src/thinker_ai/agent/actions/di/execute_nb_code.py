from __future__ import annotations

import asyncio
import base64
import re
from typing import Literal, Tuple, Dict, Any

import nbformat
from nbclient.exceptions import CellTimeoutError
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output
from rich.box import MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from thinker_ai.agent.actions import Action
from thinker_ai.common.logs import logger


class ExecuteNbCode(Action):
    """execute notebook code block, return result to llm, and display it."""

    nb: NotebookNode
    console: Console
    timeout: int = 600

    def __init__(self, nb=nbformat.v4.new_notebook(), timeout=600, **data: Any):
        super().__init__(nb=nb, console=Console(), timeout=timeout, **data)
        self.executor = ExecutePreprocessor(timeout=self.timeout, kernel_name='python3')

    def add_code_cell(self, code: str):
        self.nb.cells.append(new_code_cell(source=code))

    def add_markdown_cell(self, markdown: str):
        self.nb.cells.append(new_markdown_cell(source=markdown))

    def _display(self, code: str, language: Literal["python", "markdown"] = "python"):
        if language == "python":
            code = Syntax(code, "python", theme="paraiso-dark", line_numbers=True)
            self.console.print(code)
        elif language == "markdown":
            display_markdown(code)
        else:
            raise ValueError(f"Only support for python, markdown, but got {language}")

    def add_output_to_cell(self, cell: NotebookNode, output: str):
        """add outputs of code execution to notebook cell."""
        if "outputs" not in cell:
            cell["outputs"] = []
        else:
            cell["outputs"].append(new_output(output_type="stream", name="stdout", text=str(output)))

    def parse_outputs(self, outputs: list[Dict], keep_len: int = 2000) -> Tuple[bool, str]:
        """Parses the outputs received from notebook execution."""
        assert isinstance(outputs, list)
        parsed_output, is_success = [], True
        for i, output in enumerate(outputs):
            output_text = ""
            if output["output_type"] == "stream" and not any(
                    tag in output["text"]
                    for tag in
                    ["| INFO     | thinker_ai", "| ERROR    | thinker_ai", "| WARNING  | thinker_ai", "DEBUG"]
            ):
                output_text = output["text"]
            elif output["output_type"] == "display_data":
                if "image/png" in output["data"]:
                    self.show_bytes_figure(output["data"]["image/png"], self.interaction)
                else:
                    logger.info(
                        f"{i}th output['data'] from nbclient outputs dont have image/png, continue next output ..."
                    )
            elif output["output_type"] == "execute_result":
                output_text = output["data"]["text/plain"]
            elif output["output_type"] == "error":
                output_text, is_success = "\n".join(output["traceback"]), False

            # handle coroutines that are not executed asynchronously
            if output_text.strip().startswith("<coroutine object"):
                output_text = "Executed code failed, you need use key word 'await' to run a async code."
                is_success = False

            output_text = remove_escape_and_color_codes(output_text)
            # The useful information of the exception is at the end,
            # the useful information of normal output is at the begining.
            output_text = output_text[:keep_len] if is_success else output_text[-keep_len:]

            parsed_output.append(output_text)
        return is_success, ",".join(parsed_output)

    def show_bytes_figure(self, image_base64: str, interaction_type: Literal["ipython", "terminal"]):
        image_bytes = base64.b64decode(image_base64)
        if interaction_type == "ipython":
            from IPython.display import Image, display

            display(Image(data=image_bytes))
        else:
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            image.show()

    def is_ipython(self) -> bool:
        try:
            from IPython import get_ipython

            if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
                return True
            else:
                return False
        except NameError:
            return False

    async def run_cell(self, cell: NotebookNode) -> Tuple[bool, str]:
        """set timeout for run code.
        returns the success or failure of the cell execution, and an optional error message.
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.executor.preprocess, self.nb, {'metadata': {'path': './'}})
            outputs = self.parse_outputs(cell.outputs)
            print(f"执行结果：{str(outputs)}")
            return outputs
        except CellTimeoutError as e:
            error_msg = ("Cell execution timed out: Execution exceeded the time limit and was stopped; consider "
                         "optimizing your code for better performance.")
            self.nb.cells.remove(cell)
            return False, error_msg
        except CellExecutionError as e:
            print(f"执行异常：{str(e)}")
            self.nb.cells.remove(cell)
            return False, str(e)
        except Exception as e:
            print(f"执行异常：{str(e)}")
            self.nb.cells.remove(cell)
            return False, str(e)

    async def run(self, code: str, language: Literal["python", "markdown"] = "python") -> Tuple[str, bool]:
        """
        return the output of code execution, and a success indicator (bool) of code execution.
        """
        self._display(code, language)

        if language == "python":
            # add code to the notebook
            self.add_code_cell(code=code)

            # run code
            cell_index = len(self.nb.cells) - 1
            cell: NotebookNode=self.nb.cells[cell_index]
            success, outputs = await self.run_cell(cell)

            if "!pip" in code:
                success = False

            return outputs, success

        elif language == "markdown":
            # add markdown content to markdown cell in a notebook.
            self.add_markdown_cell(code)
            # return True, beacuse there is no execution failure for markdown cell.
            return code, True
        else:
            raise ValueError(f"Only support for language: python, markdown, but got {language}, ")

    async def reset(self):
        """reset ExecuteNbCode by creating a new notebook and executor."""
        self.nb = nbformat.v4.new_notebook()
        self.executor = ExecutePreprocessor(timeout=self.timeout, kernel_name='python3')


def remove_escape_and_color_codes(input_str: str):
    # 使用正则表达式去除jupyter notebook输出结果中的转义字符和颜色代码
    # Use regular expressions to get rid of escape characters and color codes in jupyter notebook output.
    pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
    result = pattern.sub("", input_str)
    return result


def display_markdown(content: str):
    # Use regular expressions to match blocks of code one by one.
    matches = re.finditer(r"```(.+?)```", content, re.DOTALL)
    start_index = 0
    content_panels = []
    # Set the text background color and text color.
    style = "black on white"
    # Print the matching text and code one by one.
    for match in matches:
        text_content = content[start_index: match.start()].strip()
        code_content = match.group(0).strip()[3:-3]  # Remove triple backticks

        if text_content:
            content_panels.append(Panel(Markdown(text_content), style=style, box=MINIMAL))

        if code_content:
            content_panels.append(Panel(Markdown(f"```{code_content}"), style=style, box=MINIMAL))
        start_index = match.end()

    # Print remaining text (if any).
    remaining_text = content[start_index:].strip()
    if remaining_text:
        content_panels.append(Panel(Markdown(remaining_text), style=style, box=MINIMAL))

    # Display all panels in Live mode.
    with Live(auto_refresh=False, console=Console(), vertical_overflow="visible") as live:
        live.update(Group(*content_panels))
        live.refresh()
