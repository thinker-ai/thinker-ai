from typing import Optional, List

import asynctest
from pydantic import field_validator

from thinker_ai.agent.agent import Agent
from thinker_ai.agent.llm import gpt
from langchain.pydantic_v1 import BaseModel, Field

from thinker_ai_tests.task_flow.tasks.test_result_parser import get_project_root


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


class AgentWithToolsTestCase(asynctest.TestCase):
    assistant = gpt.llm.beta.assistants.retrieve("asst_n4kxEAYXlisN7mBa9M6t7PdH")
    agent = Agent(id="001", user_id="user1", assistant=assistant, threads={}, client=gpt.llm)

    def test_chat_with_function_call(self):
        try:
            self.agent.register_function(quiz_instance.display_quiz, QuizArgs)
            generated_result = self.agent.ask(topic="quiz",
                                              content="Make a quiz with 2 questions: One open ended, one multiple choice. Then, give me feedback for the responses.")
            self.assertIsNotNone(generated_result)
            print(generated_result)
        finally:
            self.agent.remove_function()
            print(self.agent.tools)

    def test_chat_with_retrieval(self):
        file = gpt.llm.files.create(
            file=open(
                get_project_root() / "test/thinker_ai_tests/data/diy_llm.pdf",
                "rb",
            ),
            purpose="assistants",
        )
        try:
            self.agent.register_retrieval_tool()
            self.agent.register_file_id(file.id)


            generated_result = self.agent.ask(topic="file",
                                              content="解释知识库中的内容包含了什么")
            self.assertIsNotNone(generated_result)
            print(generated_result)
        finally:
            self.agent.remove_file_id(file.id)
            self.agent.remove_retrieval_tool()
            gpt.llm.files.delete(file.id)


if __name__ == '__main__':
    asynctest.main()
