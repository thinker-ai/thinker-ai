INTERPRETER_SYSTEM_MSG = """As a data scientist, you need to help user to achieve their goal step by step in a continuous Jupyter notebook. Since it is a notebook environment, don't use asyncio.run. Instead, use await if you need to call an async function."""

STRUCTUAL_PROMPT = """
# User Requirement
{user_requirement}

# Plan Status
{plan_status}

# Tool Info
{tool_info}

# Constraints
- Take on Current Task if it is in Plan Status, otherwise, tackle User Requirement directly.
- Ensure the output new code is executable in the same Jupyter notebook as the previous executed code.
- Always prioritize using pre-defined tools for the same functionality.

# Output
While some concise thoughts are helpful, code is absolutely required. Always output one and only one code block in your response. Output code in the following format:
```python
your code
```
"""

REFLECTION_SYSTEM_MSG = """You are an AI Python assistant. You will be given your previous implementation code of a task, runtime error results, and a hint to change the implementation appropriately. Write your full implementation."""

DEBUG_REFLECTION_EXAMPLE = '''
[previous impl]:
assistant:
```python
def add(a: int, b: int) -> int:
   """
   Given integers a and b, return the total value of a and b.
   """
   return a - b
```

user:
Tests failed:
assert add(1, 2) == 3 # output: -1
assert add(1, 3) == 4 # output: -2

[reflection on previous impl]:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:
def add(a: int, b: int) -> int:
   """
   Given integers a and b, return the total value of a and b.
   """
   return a + b
'''

REFLECTION_PROMPT = """
[example]
Here is an example of debugging with reflection.
{debug_example}
[/example]

[context]
{context}

[previous impl]:
{previous_impl}

[instruction]
Analyze your previous code and error in [context] step by step, provide me with improved method and code. Remember to follow [context] requirement. Don't forget to write code for steps behind the error step.
Output a json following the format:
```json
{{
    "reflection": str = "Reflection on previous implementation",
    "improved_impl": str = "Refined code after reflection.",
}}
```
"""



