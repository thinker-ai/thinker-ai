from typing import Optional, List, Dict, Any

import asynctest
from deepdiff import DeepDiff
from langchain.pydantic_v1 import BaseModel, Field
from pydantic import field_validator

from thinker_ai.agent.functions.functions_register import FunctionsRegister


# QuizArgs及其内部的QuestionModel的数据结构定义，和display_quiz实际使用的数据结构questions: List[Dict[str, Any]]定义不一致，为了
# 设计可以替换实际参数类型的条件：
# 1、数据结构的兼容性：自定义的 BaseModel 模型（例如 QuizArgs 和其内部的 QuestionModel）必须能够从实际传入的数据结构
# （例如 List[Dict[str, Any]]）中推断出来。这意味着字典中的键和模型中的字段名需要匹配，且数据类型兼容。
# 2、字段验证规则的满足：任何在 BaseModel 模型中定义的额外验证规则（例如字段长度、值范围、正则表达式等）都需要被实际数据满足。
class QuestionModel(BaseModel):
    question_text: str = Field(...)
    question_type: str = Field(..., enum=["MULTIPLE_CHOICE", "FREE_RESPONSE"])
    choices: Optional[List[str]] = None

    @classmethod
    @field_validator('choices')
    def set_choices(cls, v, values):
        if values.get('question_type') == "MULTIPLE_CHOICE" and not v:
            raise ValueError('choices must be provided for MULTIPLE_CHOICE questions')
        elif values.get('question_type') == "FREE_RESPONSE":
            return None
        return v


class QuizArgs(BaseModel):
    title: str
    questions: List[QuestionModel] = Field(
        ...,
        description="An array of questions, each with a title and potentially options (if multiple choice)."
    )


class Quiz:
    def get_mock_response_from_user_multiple_choice(self):
        return "a"

    def get_mock_response_from_user_free_response(self):
        return "I don't know."

    def display_quiz(self, title: str, questions: List[QuestionModel]):
        """
        Displays a quiz to the student, and returns the student's response. A single quiz can have multiple questions.

        :param title: The title of the quiz.
        :param questions: A list of questions, each with its own structure defining the question text, type, and choices if applicable.
        :return: A list of responses from the student.
        """
        print("Quiz:", title)
        print()
        responses = []

        for q in questions:
            print(q.question_text)
            response = ""

            # If multiple choice, print options
            if q.question_type == "MULTIPLE_CHOICE":
                for i, choice in enumerate(q.choices):
                    print(f"{i}. {choice}")
                response = self.get_mock_response_from_user_multiple_choice()

            # Otherwise, just get response
            elif q.question_type == "FREE_RESPONSE":
                response = self.get_mock_response_from_user_free_response()

            responses.append(response)
            print()

        return responses


expected_function_schema = {
    "description": "display_quiz(title, questions) - Displays a quiz to the student, and returns the student's response. A single quiz can have multiple questions.\n\n        :param title: The title of the quiz.\n        :param questions: A list of questions, each with its own structure defining the question text, type, and choices if applicable.\n        :return: A list of responses from the student.",
    "name": "display_quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "questions": {
                "type": "array",
                "description": "An array of questions, each with a title and potentially options (if multiple choice).",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_text": {"type": "string"},
                        "question_type": {
                            "type": "string",
                            "enum": ["MULTIPLE_CHOICE", "FREE_RESPONSE"],
                        },
                        "choices": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question_text", "question_type"],
                },
            },
        },
        "required": ["title", "questions"],
    },
}


class TestFunctionRegister(asynctest.TestCase):
    def test_function_call(self):
        quiz_instance = Quiz()
        functions_register = FunctionsRegister()
        functions_register.register_function(
            quiz_instance.display_quiz,
            QuizArgs)
        functions_schema = functions_register.functions_schema
        # DeepDiff能将字符串内容的格式化信息忽略，只比较内容本身，此处排除的是节点的排序差异和所有description属性中的空白字符差异
        diff = DeepDiff(expected_function_schema, functions_schema[0], exclude_regex_paths=r"root\['description'\]",
                        ignore_order=True)
        print(diff)
        # 断言没有差异
        self.assertEqual(diff, {}, "Differences found in JSON comparison")
        arguments: Dict[str, Any] = {
            'title': 'My Quiz', 'questions': [
                {'question_text': 'What is the capital of France?', 'question_type': 'FREE_RESPONSE'},
                {
                    'question_text': 'What is the largest planet in our solar system?',
                    'question_type': 'MULTIPLE_CHOICE',
                    'choices': ['Jupiter', 'Saturn', 'Earth', 'Mars']
                }
            ]
        }
        function = functions_register.get_function("display_quiz")
        if function is not None:
            result = function.invoke(arguments)
            print(result)
