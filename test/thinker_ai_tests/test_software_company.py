import pytest
from typer.testing import CliRunner

from thinker_ai.common.logs import logger
from thinker_ai.software_company import app
from thinker_ai.team import Team

runner = CliRunner()


@pytest.fixture(scope="function")
def new_filename(mocker):
    # NOTE: Mock new filename to make reproducible llm aask, should consider changing after implementing requirement segmentation
    mocker.patch("thinker_ai.utils.file_repository.FileRepository.new_filename", lambda: "20240101")
    yield mocker


@pytest.mark.asyncio
async def test_empty_team(new_filename):
    # FIXME: we're now using "thinker_ai" cli, so the entrance should be replaced instead.
    company = Team()
    history = await company.run(idea="Build a simple search system. I will upload my files later.")
    logger.info(history)


def test_software_company(new_filename):
    args = [
        "Make a cli snake game",  # idea 参数
        "--run-tests",  # 将 run_tests 设置为 True
        "--debug",  # 将 debug 设置为 True
        "--n-round", "10", # 将 n_round 设置为 10
    ]
    result = runner.invoke(app, args)
    logger.info(result)
    logger.info(result.output)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
