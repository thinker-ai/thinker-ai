import os

import fire

from thinker_ai.agent.actions.di.tool_recommend import ToolRecommender
from thinker_ai_tests.di.requirements_prompt import OPEN_ENDED_TASKS_REQUIREMENTS
from thinker_ai.configs.const import DATA_PATH
from thinker_ai.agent.roles.di.data_interpreter import DataInterpreter


# Ensure Open-Ended Tasks dataset has been downloaded before using this example.
async def main(task_name, data_dir=DATA_PATH, use_reflection=True):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/open_ended_tasks")):
        raise FileNotFoundError(f"Open-ended task dataset not found in {data_dir}.")

    requirement = OPEN_ENDED_TASKS_REQUIREMENTS[task_name].format(data_dir=data_dir)
    di = DataInterpreter(use_reflection=use_reflection, tool_recommender=ToolRecommender(tools=["<all>"]))
    await di.run(requirement)


if __name__ == "__main__":
    fire.Fire(main)
