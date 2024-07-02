# tools.py

from thinker_ai.agent.tools.tool_registry import register_tool

@register_tool()
def magic_function(arg1: str, arg2: int) -> dict:
    """
    The magic function that does something.

    Args:
        arg1 (str): ...
        arg2 (int): ...

    Returns:
        dict: ...
    """
    return {"arg1": arg1 * 3, "arg2": arg2 * 5}
