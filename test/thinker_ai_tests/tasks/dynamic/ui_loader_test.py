from fastapi import APIRouter, Depends
from starlette.responses import HTMLResponse

from thinker_ai.agent.openai_assistant_api.openai_assistant_api import OpenAiAssistantApi
from thinker_ai.tasks.dynamic.service_deployer import is_ui_exist, deploy_ui, DeployArgs

from thinker_ai.tasks.dynamic.service_loader import LoadArgs
from thinker_ai_tests.tasks.dynamic.demo_main import get_service_loader

service_loader_router = APIRouter()


@service_loader_router.get("/generate_ui", response_class=HTMLResponse)
def test_chat_with_gradio_code_generate():
    assistant = OpenAiAssistantApi.from_id(assistant_id="asst_zBrqXNoQIvnX1TyyVry9UveZ")
    try:
        assistant.register_callable(deploy_ui, DeployArgs)
        content = "请为我生成一个新的计算器，能进行加减乘除运算，我的用户id是“abc”，计算器名称叫“calculator”"
        generated_result = assistant.ask(user_id="abc",topic="加减乘除运算器", content=content)
        assert (generated_result is not None)
        print(generated_result)
        assert (is_ui_exist("calculator", "abc"), f"生成的文件不存在")
    finally:
        assistant.unload_all_callables()


@service_loader_router.get("/load_ui", response_class=HTMLResponse)
def test_chat_with_gradio_code_load():
    assistant = OpenAiAssistantApi.from_id(assistant_id="asst_zBrqXNoQIvnX1TyyVry9UveZ")
    try:
        assistant.register_callable(get_service_loader().load_ui_and_show, LoadArgs)
        content = "请加载名称为“calculator”的计算器并展示给id是“abc”的用户"
        generated_result = assistant.ask(user_id="abc",topic="加减乘除运算器", content=content)
        assert (generated_result is not None)
    finally:
        assistant.unload_all_callables()
