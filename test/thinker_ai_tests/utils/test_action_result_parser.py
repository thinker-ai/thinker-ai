import json
from json import JSONDecodeError
from typing import List, Tuple, Dict

import pytest

from thinker_ai.utils.code_parser import CodeParser
from thinker_ai.utils.text_parser import TextParser


def test_parse_blocks():
    test_text = "##block1\nThis is block user_1.\n##block2\nThis is block 2."
    expected_result = {'block1': 'This is block user_1.', 'block2': 'This is block 2.'}
    assert TextParser.parse_blocks(test_text) == expected_result


def test_parse_code():
    test_text = "```python\nprint('Hello, world!')```"
    expected_result = "print('Hello, world!')"
    assert TextParser.parse_code(test_text, 'python') == expected_result

    with pytest.raises(Exception):
        TextParser.parse_code(test_text, 'java')


def test_parse_code_2():
    rsp = """```json
{
    "reflection": "The previous implementation encountered an error due to an unterminated string. This error typically occurs when a string literal is not properly closed with a matching quote. The error message indicates that the issue is at line 3, column 22. This suggests that there might be a missing quote in the string definition. Additionally, the context requires us to train a model to predict passenger survival using the preprocessed data. We need to ensure that the code is executable in the same Jupyter notebook as the previous executed code and prioritize using pre-defined tools for the same functionality.",
    "improved_impl": "```python\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n\n# Assuming train_df_copy and eval_df_copy are already defined from previous steps\n\n# Splitting the data into features and target\nX = train_df_copy.drop(columns=['Survived'])\ny = train_df_copy['Survived']\n\n# Splitting the data into training and validation sets\nX_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Training a RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\n\n# Making predictions on the validation set\ny_pred = model.predict(X_val)\n\n# Calculating the accuracy\naccuracy = accuracy_score(y_val, y_pred)\nprint(f'Validation Accuracy: {accuracy}')\n\n# Making predictions on the evaluation set\neval_X = eval_df_copy\neval_predictions = model.predict(eval_X)\nprint(f'Evaluation Predictions: {eval_predictions}')\n```"
}
```"""
    json_str = CodeParser.parse_code(block=None, text=rsp)
    try:
        reflection = json.loads(json_str)
        assert reflection["improved_impl"] == "python\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n\n# Assuming train_df_copy and eval_df_copy are already defined from previous steps\n\n# Splitting the data into features and target\nX = train_df_copy.drop(columns=['Survived'])\ny = train_df_copy['Survived']\n\n# Splitting the data into training and validation sets\nX_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Training a RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\n\n# Making predictions on the validation set\ny_pred = model.predict(X_val)\n\n# Calculating the accuracy\naccuracy = accuracy_score(y_val, y_pred)\nprint(f'Validation Accuracy: {accuracy}')\n\n# Making predictions on the evaluation set\neval_X = eval_df_copy\neval_predictions = model.predict(eval_X)\nprint(f'Evaluation Predictions: {eval_predictions}')\n"
    except JSONDecodeError as err:
        print(err.args)
        assert False


def normalize_code(code: str) -> str:
    """
    Normalize the code string by stripping leading/trailing whitespace
    and removing redundant blank lines and uniforming indentation.
    """
    # Split the code into lines
    lines = code.splitlines()

    # Strip leading and trailing whitespace from each line
    lines = [line.strip() for line in lines]

    # Remove leading and trailing blank lines
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop(-1)

    # Rejoin the lines into a single string with normalized indentation
    return "\n".join(lines)

def test_parse_python_code():
    expected_result = "print('Hello, world!')"
    assert TextParser.parse_python_code("```python\nprint('Hello, world!')```") == expected_result
    assert TextParser.parse_python_code("```python\nprint('Hello, world!')") == expected_result
    assert TextParser.parse_python_code("print('Hello, world!')") == expected_result
    assert TextParser.parse_python_code("print('Hello, world!')```") == expected_result
    assert TextParser.parse_python_code("print('Hello, world!')```") == expected_result
    expected_result = "print('```Hello, world!```')"
    assert TextParser.parse_python_code("```python\nprint('```Hello, world!```')```") == expected_result
    assert TextParser.parse_python_code(
        "The code1 is: ```python\nprint('```Hello, world!```')```") == expected_result
    assert TextParser.parse_python_code(
        "xxx.\n```python\nprint('```Hello, world!```')```\nxxx") == expected_result

    with pytest.raises(ValueError):
        TextParser.parse_python_code("xxx =")


def test_parse_str():
    test_text = "name = 'Alice'"
    expected_result = 'Alice'
    assert TextParser.parse_str(test_text) == expected_result


def test_parse_file_list():
    test_text = "files=['file1', 'file2', 'file3']"
    expected_result = ['file1', 'file2', 'file3']
    assert TextParser.parse_list(test_text) == expected_result

    with pytest.raises(Exception):
        TextParser.parse_list("wrong_input")


def test_parse_data():
    test_data = "##block1\n```python\nprint('Hello, world!')```\n##block2\nfiles=['file1', 'file2', 'file3']"
    expected_result = {'block1': "print('Hello, world!')", 'block2': ['file1', 'file2', 'file3']}
    assert TextParser.parse_data(test_data) == expected_result


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
        ("Create Blog Post", "User creates a new blog post", "P0"),
        ("Read Blog Post", "User reads an existing blog post", "P0"),
    ]
    assert TextParser.parse_list(test_data) == expected_result


def test_parse_dict():
    test_data = """
    ## Use Case Detail
    ```python
{
    "Create User Information": {
        "Success Path": [
            "user_1. Actor inputs personal information (name, age, email)",
            "2. System validates the input and creates a new user record, returns a success message"
        ],
        "Failure Path": [
            "2a. System validates input error, prompts 'Invalid input'",
            "2a1. Actor re-enters, return to step user_1"
        ],
        "I/O and Validation Rules": {
            "user_1. input": ["name (string, not null)", "age (integer, not null)", "email (string, not null)"],
            "2. output": ["success message (string)"]
        }
    },
    "Read User Information": {
        "Success Path": [
            "user_1. Actor requests to read personal information",
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
    expected_result = {"Use Case Detail": {
        "Create User Information": {
            "Success Path": [
                "user_1. Actor inputs personal information (name, age, email)",
                "2. System validates the input and creates a new user record, returns a success message"
            ],
            "Failure Path": [
                "2a. System validates input error, prompts 'Invalid input'",
                "2a1. Actor re-enters, return to step user_1"
            ],
            "I/O and Validation Rules": {
                "user_1. input": ["name (string, not null)", "age (integer, not null)", "email (string, not null)"],
                "2. output": ["success message (string)"]
            }
        },
        "Read User Information": {
            "Success Path": [
                "user_1. Actor requests to read personal information",
                "2. System retrieves and returns the user's personal information"
            ],
            "Failure Path": [],
            "I/O and Validation Rules": {
                "2. output": ["name (string)", "age (integer)", "email (string)"]
            }
        }
    }
    }
    OUTPUT_MAPPING = {
        "Use Case Detail": (Dict, ...),
    }
    result = TextParser.parse_data_with_mapping(test_data, OUTPUT_MAPPING)
    assert result.dict() == expected_result


def test_all():
    data = """
## Original Idea
The original idea is to develop a system that can effectively manage the user requirements in a structured manner. The system should be able to identify user needs and provide a structured representation of user requirements. The system should be user-friendly and efficient in managing the requirements.

## Business Analysis
To achieve the business objectives, the system should have the following features:
user_1. Requirement Identification: The system should be able to identify the user needs effectively.
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
    OUTPUT_MAPPING = {
        "Original Idea": (str, ...),
        "Business Analysis": (str, ...),
        "Business Entity": (str, ...),
        "Business Process": (str, ...),
        "Use Case list": (List[Tuple[str, str, str]], ...),
        "Use Case Detail": (Dict, ...),
        "Anything UNCLEAR": (str, ...),
    }
    instruct_content = TextParser.parse_data_with_mapping(data, OUTPUT_MAPPING)
    print(instruct_content)
