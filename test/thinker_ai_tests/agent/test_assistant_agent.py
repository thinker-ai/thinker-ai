from typing import Optional, List, cast

import asynctest
import numpy as np
import pandas as pd
from pydantic import field_validator

from thinker_ai.agent.assistant_agent import AssistantAgent
from langchain.pydantic_v1 import BaseModel, Field

from thinker_ai.agent.provider.llm import open_ai


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


class AgentWithToolsTestCase(asynctest.TestCase):
    agent = AssistantAgent.from_id("asst_zBrqXNoQIvnX1TyyVry9UveZ")

    def test_chat_with_function_call(self):
        try:
            self.agent.register_function(quiz_instance.display_quiz, QuizArgs)
            generated_result = self.agent.ask(topic="quiz",
                                              content="Make a quiz with 2 questions: One open ended, one multiple choice. Then, give me feedback for the responses.")
            self.assertIsNotNone(generated_result)
            print(generated_result)
        finally:
            self.agent.remove_functions()
            print(self.agent.tools)

    def test_chat_with_code_interpreter_1(self):
        ask= "Category_A的自变量是1到100的自然数，求Category_A数据的函数表达式，并绘制这个函数的图形,输出得到这个函数的python源代码。"
        self.chat_with_code_interpreter(ask)

    def test_chat_with_code_interpreter_2(self):
        ask= "Category_A数据是Category_B数据的自变量，求它们的函数关系，并绘制这个函数的图形,输出得到这个函数的python源代码。"
        self.chat_with_code_interpreter(ask)

    def chat_with_code_interpreter(self, ask):
        data = create_correlated_data()
        try:
            json_data = data.to_json(orient='records')
            self.agent.set_instructions("你是一个数学老师，负责回答数学问题")
            self.agent.register_code_interpreter()
            generated_result = self.agent.ask(topic="math_tutor",
                                              content=f"问题：{ask}。具体数据如下：{json_data}")
            self.assertIsNotNone(generated_result)
            img_index = 0
            for result in generated_result:
                if result.get("text"):
                    print(result.get("text"))
                if result.get("image_file"):
                    with open(f"data/math_tutor-image-{img_index}.png", "wb") as file:
                        file.write(result.get("image_file"))
                        img_index = img_index + 1
        finally:
            self.agent.remove_functions()
            print(self.agent.tools)

    def test_chat_with_file_search(self):
        file = open_ai.client.files.create(
            file=open(
                "data/diy_llm.pdf",
                "rb",
            ),
            purpose="assistants",
        )
        try:
            self.agent.register_file_search()
            self.agent.register_file_id(file.id)

            generated_result = self.agent.ask(topic="file",
                                              content="解释知识库中的内容包含了什么")
            self.assertIsNotNone(generated_result)
            print(generated_result)
        finally:
            self.agent.remove_file_id(file.id)
            self.agent.remove_file_search()
            open_ai.client.files.delete(file.id)


if __name__ == '__main__':
    asynctest.main()
