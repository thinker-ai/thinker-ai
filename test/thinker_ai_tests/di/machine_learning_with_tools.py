import asyncio
import json

from thinker_ai.agent.provider.schema import Message
from thinker_ai.agent.roles.di.data_interpreter import DataInterpreter


async def main(content: str):
    role = DataInterpreter(use_reflection=True, tools=["<all>"])
    await role.run(content)


if __name__ == "__main__":
    data_path = "../data/titanic"
    train_path = f"{data_path}/train.csv"
    eval_path = f"{data_path}/test.csv"
    requirement={
        "name" : "titanic_survival",
        "instruction" : f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report accuracy on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'."
    }
    asyncio.run(main(json.dumps(requirement)))
