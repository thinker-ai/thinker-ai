import asyncio

from thinker_ai.common.logs import logger
from thinker_ai.agent.roles.di.data_interpreter import DataInterpreter
from thinker_ai.utils.recovery_util import save_history


async def main(requirement: str = ""):
    di = DataInterpreter()
    rsp = await di.run(requirement)
    logger.info(rsp)
    save_history(role=di)


if __name__ == "__main__":
    requirement = "Run data analysis on sklearn Iris dataset, include a plot"
    asyncio.run(main(requirement))
