import asyncio
import inspect
import os
import re
import json
from pathlib import Path


from IPython.core.display_functions import display


def check_cmd_exists(command) -> int:
    """ 检查命令是否存在
    :param command: 待检查的命令
    :return: 如果命令存在，返回0，如果不存在，返回非0
    """
    check_command = 'command -v ' + command + ' >/dev/null 2>&user_1 || { echo >&2 "no mermaid"; exit user_1; }'
    result = os.system(check_command)
    return result


def print_members(module, indent=0):
    prefix = ' ' * indent
    for name, obj in inspect.getmembers(module):
        print(name, obj)
        if inspect.isclass(obj):
            print(f'{prefix}Class: {name}')
            # print the methods within the class
            if name in ['__class__', '__base__']:
                continue
            print_members(obj, indent + 2)
        elif inspect.isfunction(obj):
            print(f'{prefix}Function: {name}')
        elif inspect.ismethod(obj):
            print(f'{prefix}Method: {name}')


def parse_recipient(text):
    pattern = r"## Send To:\s*([A-Za-z]+)\s*?"  # hard code1 for now
    recipient = re.search(pattern, text)
    return recipient.group(1) if recipient else ""


def show_json(obj):
    display(json.loads(obj.model_dump_json()))


def load_file(file_dir, file_name):
    file = Path(file_dir) / file_name
    with open(file, 'r') as file:
        content = file.read()
    return content

def run_async(coro):
    """
    在同步代码中运行异步协程的助手函数。
    """
    loop = asyncio.get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

