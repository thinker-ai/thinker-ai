from fastapi import APIRouter
from starlette.responses import HTMLResponse

from thinker_ai.agent.openai_assistant_agent import AssistantAgent
from thinker_ai.tasks.dynamic.service_deployer import is_service_exist, deploy_service
from langchain.pydantic_v1 import BaseModel, Field

from thinker_ai.tasks.dynamic.service_loader import load_service_and_push_to_user

service_loader_router = APIRouter()
agent = AssistantAgent.from_id("asst_zBrqXNoQIvnX1TyyVry9UveZ")

class DeployArgs(BaseModel):
    name: str = Field(
        ...,
        description="应用名称，即是gradio文件名，也是代码中最后一句创建的Blocks实例的名称"
    )
    gradio_code: str = Field(
        ...,
        description="gradio代码内容，所有import语句只能是局部作用域，这是为了回避gradio的bug"
    )
    user_id: str = Field(
        ...,
        description="用户id"
    )


class LoadArgs(BaseModel):
    name: str = Field(
        ...,
        description="应用名称，即是gradio文件名"
    )
    user_id: str = Field(
        ...,
        description="用户id"
    )


@service_loader_router.get("/code_generate", response_class=HTMLResponse)
def test_chat_with_gradio_code_generate():
    try:
        agent.register_function(deploy_service, DeployArgs)
        content = "请为我生成一个新的计算器，能进行加减乘除运算，我的用户id是“abc”，计算器名称叫“calculator”"
        generated_result = agent.ask(topic="加减乘除运算器", content=content)
        assert(generated_result is not None)
        print(generated_result)
        assert (is_service_exist("calculator", "abc"), f"生成的文件不存在")
    finally:
        agent.remove_functions()


@service_loader_router.get("/code_load", response_class=HTMLResponse)
def test_chat_with_gradio_code_load():
    try:
        agent.register_function(load_service_and_push_to_user, LoadArgs)
        content = "请给我提供一个已有的计算器，我的用户id是“abc”，计算器名称叫“calculator”"
        generated_result = agent.ask(topic="加减乘除运算器", content=content)
        assert(generated_result is not None)
    finally:
        agent.remove_functions()
