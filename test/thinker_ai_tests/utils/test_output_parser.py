import ast
from typing import List, Tuple, Dict

import pytest

from thinker_ai.action.action_output import ActionOutput
from thinker_ai.utils.output_parser import OutputParser

OUTPUT_MAPPING = {
    "Original Idea": (str, ...),
    "Business Analysis": (str, ...),
    "Business Entity": (str, ...),
    "Business Process": (str, ...),
    "Use Case list": (List[Tuple[str,str,str]], ...),
    "Use Case Detail": (Dict, ...),
    "Anything UNCLEAR": (str, ...),
}
def test_parse_blocks():
    test_text = "##block1\nThis is block 1.\n##block2\nThis is block 2."
    expected_result = {'block1': 'This is block 1.', 'block2': 'This is block 2.'}
    assert OutputParser.parse_blocks(test_text) == expected_result


def test_parse_code():
    test_text = "```python\nprint('Hello, world!')```"
    expected_result = "print('Hello, world!')"
    assert OutputParser.parse_code(test_text, 'python') == expected_result

    with pytest.raises(Exception):
        OutputParser.parse_code(test_text, 'java')


def test_parse_python_code():
    expected_result = "print('Hello, world!')"
    assert OutputParser.parse_python_code("```python\nprint('Hello, world!')```") == expected_result
    assert OutputParser.parse_python_code("```python\nprint('Hello, world!')") == expected_result
    assert OutputParser.parse_python_code("print('Hello, world!')") == expected_result
    assert OutputParser.parse_python_code("print('Hello, world!')```") == expected_result
    assert OutputParser.parse_python_code("print('Hello, world!')```") == expected_result
    expected_result = "print('```Hello, world!```')"
    assert OutputParser.parse_python_code("```python\nprint('```Hello, world!```')```") == expected_result
    assert OutputParser.parse_python_code("The code is: ```python\nprint('```Hello, world!```')```") == expected_result
    assert OutputParser.parse_python_code("xxx.\n```python\nprint('```Hello, world!```')```\nxxx") == expected_result

    with pytest.raises(ValueError):
        OutputParser.parse_python_code("xxx =")


def test_parse_str():
    test_text = "name = 'Alice'"
    expected_result = 'Alice'
    assert OutputParser.parse_str(test_text) == expected_result


def test_parse_file_list():
    test_text = "files=['file1', 'file2', 'file3']"
    expected_result = ['file1', 'file2', 'file3']
    assert OutputParser.parse_list(test_text) == expected_result

    with pytest.raises(Exception):
        OutputParser.parse_list("wrong_input")


def test_parse_data():
    test_data = "##block1\n```python\nprint('Hello, world!')```\n##block2\nfiles=['file1', 'file2', 'file3']"
    expected_result = {'block1': "print('Hello, world!')", 'block2': ['file1', 'file2', 'file3']}
    assert OutputParser.parse_data(test_data) == expected_result


def test_parse_tuple_list():
    test_data = """
    ## Use Case list
    ```python
    [
        ("Create Blog Post","User creates a new blog post","P0"),
        ("Read Blog Post","User reads an existing blog post","P0"),
    ]
    ```"""
    expected_result = [
        ("Create Blog Post","User creates a new blog post","P0"),
        ("Read Blog Post","User reads an existing blog post","P0"),
    ]
    assert OutputParser.parse_list(test_data) == expected_result


def test_parse_dict():
    test_data = """
    ## Use Case Detail
    ```python
{
    "Create User Information": {
        "Success Path": [
            "1. Actor inputs personal information (name, age, email)",
            "2. System validates the input and creates a new user record, returns a success message"
        ],
        "Failure Path": [
            "2a. System validates input error, prompts 'Invalid input'",
            "2a1. Actor re-enters, return to step 1"
        ],
        "I/O and Validation Rules": {
            "1. input": ["name (string, not null)", "age (integer, not null)", "email (string, not null)"],
            "2. output": ["success message (string)"]
        }
    },
    "Read User Information": {
        "Success Path": [
            "1. Actor requests to read personal information",
            "2. System retrieves and returns the user's personal information"
        ],
        "Failure Path": [],
        "I/O and Validation Rules": {
            "2. output": ["name (string)", "age (integer)", "email (string)"]
        }
    }
}
    ```
    """
    expected_result = {"Use Case Detail":{
            "Create User Information": {
                "Success Path": [
                    "1. Actor inputs personal information (name, age, email)",
                    "2. System validates the input and creates a new user record, returns a success message"
                ],
                "Failure Path": [
                    "2a. System validates input error, prompts 'Invalid input'",
                    "2a1. Actor re-enters, return to step 1"
                ],
                "I/O and Validation Rules": {
                    "1. input": ["name (string, not null)", "age (integer, not null)", "email (string, not null)"],
                    "2. output": ["success message (string)"]
                }
            },
            "Read User Information": {
                "Success Path": [
                    "1. Actor requests to read personal information",
                    "2. System retrieves and returns the user's personal information"
                ],
                "Failure Path": [],
                "I/O and Validation Rules": {
                    "2. output": ["name (string)", "age (integer)", "email (string)"]
                }
            }
       }
    }
    assert OutputParser.parse_data_with_mapping(test_data,OUTPUT_MAPPING) == expected_result


def test_all():
    data = """
    #### Original Idea
The original idea is to develop a system that can effectively manage the user requirements in a structured manner. The system should be able to identify user needs and provide a structured representation of user requirements. The system should be user-friendly and efficient in managing the requirements.

## Business Analysis
To achieve the business objectives, the system should have the following features:
1. Requirement Identification: The system should be able to identify the user needs effectively.
2. Structured Representation: The system should provide a structured representation of user requirements.
3. User-friendly Interface: The system should have a user-friendly interface for easy navigation and usage.
4. Efficient Management: The system should be efficient in managing the user requirements.

## Business Entity
```mermaid
erDiagram
    SYSTEM ||--o{ USER : manages
    REQUIREMENT ||--o{ SYSTEM : is_managed_by
```

## Business Process
```mermaid
flowchart TB
    User-->|Identify Requirement|System
    System-->|Structured Representation|User
    User-->|Manage Requirement|System
    System-->|Provide Feedback|User
```

## Use Case list
```python
[
    ("Identify Requirement","User identifies the requirement","P0"),
    ("Structured Representation","System provides a structured representation of the requirement","P1"),
    ("Manage Requirement","User manages the requirement","P0"),
    ("Provide Feedback","System provides feedback to the user","P2")
]
```

## Use Case Detail
```python
{
    "Identify Requirement": {
        "Success Path": [
            "User inputs the requirement",
            "System identifies the requirement and returns a successful result"
        ],
        "Failure Path": [
            "System fails to identify the requirement, prompts error",
            "User re-enters the requirement"
        ],
        "I/O and Validation Rules": [
            {"input": ["requirement", "string", "not null"]},
            {"output": ["result", "string"]}
        ]
    },
    "Structured Representation": {
        "Success Path": [
            "User inputs the requirement",
            "System provides a structured representation of the requirement and returns a successful result"
        ],
        "Failure Path": [
            "System fails to provide a structured representation, prompts error",
            "User re-enters the requirement"
        ],
        "I/O and Validation Rules": [
            {"input": ["requirement", "string", "not null"]},
            {"output": ["result", "string"]}
        ]
    },
    "Manage Requirement": {
        "Success Path": [
            "User inputs the requirement",
            "System manages the requirement and returns a successful result"
        ],
        "Failure Path": [
            "System fails to manage the requirement, prompts error",
            "User re-enters the requirement"
        ],
        "I/O and Validation Rules": [
            {"input": ["requirement", "string", "not null"]},
            {"output": ["result", "string"]}
        ]
    },
    "Provide Feedback": {
        "Success Path": [
            "User inputs the requirement",
            "System provides feedback and returns a successful result"
        ],
        "Failure Path": [
            "System fails to provide feedback, prompts error",
            "User re-enters the requirement"
        ],
        "I/O and Validation Rules": [
            {"input": ["requirement", "string", "not null"]},
            {"output": ["result", "string"]}
        ]
    }
}
```

## Anything UNCLEAR
There are no unclear points.
    """

    output_class = ActionOutput.create_model_class("prd", OUTPUT_MAPPING)
    parsed_data = OutputParser.parse_data_with_mapping(data, OUTPUT_MAPPING)
    instruct_content = output_class(**parsed_data)
    print(instruct_content)