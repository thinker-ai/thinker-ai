import pytest

from thinker_ai.agent.actions.di.task import AskReview


@pytest.mark.asyncio
async def test_ask_review(mocker):
    mock_review_input = "confirm"
    mocker.patch("builtins.input", return_value=mock_review_input)
    rsp, confirmed = await AskReview().run()
    assert rsp == mock_review_input
    assert confirmed
