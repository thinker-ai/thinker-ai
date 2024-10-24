from typing import Optional, List, cast
import unittest
import numpy as np
import pandas as pd
from pydantic import field_validator

from thinker_ai.agent.openai_assistant_api import openai
from thinker_ai.agent.openai_assistant_api.openai_assistant_api import OpenAiAssistantApi
from langchain.pydantic_v1 import BaseModel, Field


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


quiz_instance = Quiz()


def create_correlated_data():
    np.random.seed(42)  # Seed for reproducibility
    days = np.arange(1, 101)  # Time feature from day user_1 to 100
    dates = pd.date_range(start='2021-01-01', periods=100)  # Generating dates

    # Creating complex relationships
    category_A = 0.05 * days ** 2 + 2 * days + 5 + np.random.normal(0, 10, days.size)  # Quadratic with noise
    category_B = 100 * np.sin(0.1 * days) + 50 + np.random.normal(0, 5, days.size)  # Sinusoidal with noise

    # Creating a DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Category_A': category_A,
        'Category_B': category_B
    })
    pd.set_option('display.max_rows', None)
    return data


class AgentWithToolsTestCase(unittest.IsolatedAsyncioTestCase):
    callable_name = openai.callables_register.register_callable(quiz_instance.display_quiz, QuizArgs)
    assistant = OpenAiAssistantApi.from_id(assistant_id="asst_jVuGNopfYaibKPYdvNAVEnh2")
    assistant.load_callables({callable_name})
    async def test_chat_with_function_call(self):
        try:
            generated_result = self.assistant.ask(user_id="abc",topic="quiz",
                                                  content="Make a quiz with 2 questions: One open ended, one multiple choice. Then, give me feedback for the responses.")
            self.assertIsNotNone(generated_result)
            print(generated_result)
        finally:
            self.assistant.unload_all_callables()
            print(self.assistant.tools)

    async def test_chat_with_code_interpreter_1(self):
        ask = "Category_A的自变量是1到100的自然数，求Category_A数据的函数表达式，并绘制这个函数的图形,输出得到这个函数的python源代码。"
        await self.chat_with_code_interpreter(ask)

    async def test_chat_with_code_interpreter_2(self):
        ask = "Category_A数据是Category_B数据的自变量，求它们的函数关系，并绘制这个函数的图形,输出得到这个函数的python源代码。"
        await self.chat_with_code_interpreter(ask)

    async def chat_with_code_interpreter(self, ask):
        data = create_correlated_data()
        try:
            json_data = data.to_json(orient='records')
            self.assistant.set_instructions("你是一个数学老师，负责回答数学问题")
            self.assistant.register_code_interpreter()
            generated_result = self.assistant._ask_for_messages(user_id="abc",topic_name="math_tutor",
                                                                content=f"问题：{ask}。具体数据如下：{json_data}")
            self.assertIsNotNone(generated_result)
            results = {}
            text_index = 0
            img_index = 0
            for content in generated_result.content:
                if content.type == "text":
                    results[f"text_{text_index}"] = self.assistant._do_with_text_result(content.text)
                    text_index += 1
                if content.type == "image_file":
                    results[f"image_file_{img_index}"] = self.assistant._do_with_image_result(content.image_file)
                    img_index += 1
            for key,content in results.items():
                if key.startswith("text_"):
                    print(content)
                if key.startswith("image_file_"):
                    with open(f"data/math_tutor-{key}.png", "wb") as file:
                        file.write(content)
        finally:
            self.assistant.unload_all_callables()
            print(self.assistant.tools)

    async def test_chat_with_file_search(self):
        file = openai.client.files.create(
            file=open(
                "data/diy_llm.pdf",
                "rb",
            ),
            purpose="assistants",
        )
        try:
            self.assistant.register_file_search()
            self.assistant.register_file_id(file.id)

            generated_result = self.assistant.ask(user_id="abc",topic="file",
                                                  content="解释知识库中的内容包含了什么")
            self.assertIsNotNone(generated_result)
            print(generated_result)
        finally:
            self.assistant.remove_file_id(file.id)
            self.assistant.remove_file_search()
            openai.client.files.delete(file.id)


if __name__ == '__main__':
    unittest.main()
