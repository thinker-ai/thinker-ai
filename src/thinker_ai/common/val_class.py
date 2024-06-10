import importlib
import inspect
import os
import re
from typing import Callable, Any, Union


def get_class_name(cls) -> str:
    """Return class name"""
    return f"{cls.__module__}.{cls.__name__}"


def any_to_str(val: Any) -> str:
    """Return the class name or the class name of the object, or 'val' if it's a string type."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)


def any_to_name(val):
    """
    Convert a value to its name by extracting the last part of the dotted path.
    """
    return any_to_str(val).split(".")[-1]


def any_to_str_set(val) -> set:
    """Convert any type to string set."""
    res = set()

    # Check if the value is iterable, but not a string (since strings are technically iterable)
    if isinstance(val, (dict, list, set, tuple)):
        # Special handling for dictionaries to iterate over values
        if isinstance(val, dict):
            val = val.values()

        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))

    return res


def print_members(module, indent=0):
    """
    https://stackoverflow.com/questions/1796180/how-can-i-get-a-list-of-all-classes-within-current-module-in-python
    """
    prefix = " " * indent
    for name, obj in inspect.getmembers(module):
        print(name, obj)
        if inspect.isclass(obj):
            print(f"{prefix}Class: {name}")
            # print the methods within the class
            if name in ["__class__", "__base__"]:
                continue
            print_members(obj, indent + 2)
        elif inspect.isfunction(obj):
            print(f"{prefix}Function: {name}")
        elif inspect.ismethod(obj):
            print(f"{prefix}Method: {name}")


def get_function_schema(func: Callable) -> dict[str, Union[dict, Any, str]]:
    sig = inspect.signature(func)
    parameters = sig.parameters
    return_type = sig.return_annotation
    param_schema = {name: parameter.annotation for name, parameter in parameters.items()}
    return {"input_params": param_schema, "return_type": return_type, "func_desc": func.__doc__, "func": func}

def parse_recipient(text):
    # FIXME: use ActionNode instead.
    pattern = r"## Send To:\s*([A-Za-z]+)\s*?"  # hard code for now
    recipient = re.search(pattern, text)
    if recipient:
        return recipient.group(1)
    pattern = r"Send To:\s*([A-Za-z]+)\s*?"
    recipient = re.search(pattern, text)
    if recipient:
        return recipient.group(1)
    return ""


def remove_comments(code_str: str) -> str:
    """Remove comments from code."""
    pattern = r"(\".*?\"|\'.*?\')|(\#.*?$)"

    def replace_func(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    clean_code = re.sub(pattern, replace_func, code_str, flags=re.MULTILINE)
    clean_code = os.linesep.join([s.rstrip() for s in clean_code.splitlines() if s.strip()])
    return clean_code




def is_coroutine_func(func: Callable) -> bool:
    return inspect.iscoroutinefunction(func)


def import_class(class_name: str, module_name: str) -> type:
    module = importlib.import_module(module_name)
    a_class = getattr(module, class_name)
    return a_class


def import_class_inst(class_name: str, module_name: str, *args, **kwargs) -> object:
    a_class = import_class(class_name, module_name)
    class_inst = a_class(*args, **kwargs)
    return class_inst
