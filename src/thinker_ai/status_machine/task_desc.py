from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel

from thinker_ai.status_machine.task_type import (
    DATA_PREPROCESS_PROMPT,
    EDA_PROMPT,
    FEATURE_ENGINEERING_PROMPT,
    IMAGE2WEBPAGE_PROMPT,
    MODEL_EVALUATE_PROMPT,
    MODEL_TRAIN_PROMPT, STATE_MACHINE_PLAN_PROMPT,
)


class TaskDesc(BaseModel):
    map: dict[str, Any]

    @property
    def id(self) -> str:
        return self.map.get('id')

    @property
    def instruction(self) -> str:
        return self.map.get('instruction')

    @property
    def plan(self) -> list[dict]:
        return self.map.get('plan')

    @property
    def guidance(self) -> str:
        GUIDANCE_TEMP: str = """
        Guidance:
             Write complete code for 'Current Task'. And avoid duplicating code from 'Finished Tasks', such as repeated import of packages, reading data, etc.Specifically，{guidance}
        """
        task_type = TaskType.get_type(self.map.get('type'))
        return GUIDANCE_TEMP.format(guidance=task_type.guidance) if task_type else ""

    def to_string(self) -> str:
        FOMART: str = """
        Describe:
            id: {id}
            instruction:{instruction}
        """
        result = FOMART.format(id=self.id, instruction=self.instruction)
        if self.plan:
            PLAN_FOMART = """
            Plan:
                {plan}
            """
            result += PLAN_FOMART.format(plan=self.plan)
        if self.guidance:
            result += self.guidance
        return result


class TaskTypeDef(BaseModel):
    name: str
    desc: str = ""
    guidance: str = ""


class TaskType(Enum):
    """By identifying specific types of tasks, we can inject human priors (guidance) to help task solving"""
    # PLAN = TaskTypeDef(
    #     name="plan",
    #     desc="Make a plan, decompose the given task according to the Single Level of Abstraction (SLOA) principle",
    #     guidance=PLAN_PROMPT,
    # )

    STATE_MACHINE_PLAN = TaskTypeDef(
        name="state machine plan",
        desc="Generate state machines based on instructions",
        guidance=STATE_MACHINE_PLAN_PROMPT,
    )

    EDA = TaskTypeDef(
        name="eda",
        desc="For performing exploratory data analysis",
        guidance=EDA_PROMPT,
    )
    DATA_PREPROCESS = TaskTypeDef(
        name="data preprocessing",
        desc="For preprocessing dataset in a data analysis or machine learning task ONLY,"
             "general data operation doesn't fall into this type",
        guidance=DATA_PREPROCESS_PROMPT,
    )
    FEATURE_ENGINEERING = TaskTypeDef(
        name="feature engineering",
        desc="Only for creating new columns for input data.",
        guidance=FEATURE_ENGINEERING_PROMPT,
    )
    MODEL_TRAIN = TaskTypeDef(
        name="model train",
        desc="Only for training model.",
        guidance=MODEL_TRAIN_PROMPT,
    )
    MODEL_EVALUATE = TaskTypeDef(
        name="model evaluate",
        desc="Only for evaluating model.",
        guidance=MODEL_EVALUATE_PROMPT,
    )
    IMAGE2WEBPAGE = TaskTypeDef(
        name="image2webpage",
        desc="For converting image into webpage code.",
        guidance=IMAGE2WEBPAGE_PROMPT,
    )
    OTHER = TaskTypeDef(name="other", desc="Any tasks not in the defined categories")

    # Legacy TaskType to support tool recommendation using type match. You don't need to define task types if you have no human priors to inject.
    TEXT2IMAGE = TaskTypeDef(
        name="text2image",
        desc="Related to text2image, image2image using stable diffusion model.",
    )
    WEBSCRAPING = TaskTypeDef(
        name="web scraping",
        desc="For scraping data from web pages.",
    )
    EMAIL_LOGIN = TaskTypeDef(
        name="email login",
        desc="For logging to an email.",
    )

    @property
    def type_name(self):
        return self.value.name

    @classmethod
    def get_type(cls, type_name: str) -> Optional[TaskTypeDef]:
        for member in cls:
            if member.type_name == type_name:
                return member.value
        return None


class PlanStatus(BaseModel):
    code_written: dict[str,str]
    exec_results: dict[str,str]

    def to_string(self) -> str:
        PLAN_STATUS = """
    ## Finished Tasks
    ### tasks id list
    {finished_tasks}
    ### finished code
    ```python
    {code_written}
    ```
    ### execution result
    {exec_results}
        """
        # combine components in a prompt
        prompt = PLAN_STATUS.format(
            finished_tasks="\n".join(self.code_written.keys()),  # Use "\n" for line breaks
            code_written="\n".join(self.code_written.values()),  # Use "\n" for line breaks
            exec_results="\n".join(self.exec_results.values()),  # Use "\n" for line breaks
        )
        return prompt
