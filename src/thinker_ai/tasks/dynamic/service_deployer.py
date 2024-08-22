import os
from typing import Dict, Optional
from langchain.pydantic_v1 import BaseModel, Field
from thinker_ai.tasks.dynamic.service import Service, get_deploy_path


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


def deploy_ui(name, gradio_code, user_id: Optional[str]) -> str:
    """
    保存一段已生成的gradio代码
    :param name: gradio代码文件名.
    :param user_id: 用户id
    :return: 文件的保存路径.
    """
    deploy_path = get_deploy_path(name, user_id)
    os.makedirs(os.path.dirname(deploy_path), exist_ok=True)
    with open(deploy_path, 'w') as file:
        file.write(gradio_code)
    return deploy_path


def is_ui_exist(name, user_id: Optional[str]) -> bool:
    """
    判断一份gradio代码是否存在
    :param name: gradio代码文件名.
    :param user_id: 用户id
    """
    deploy_path = get_deploy_path(name, user_id)
    try:
        with open(deploy_path, 'r') as file:
            return file.readable()
    except FileNotFoundError:
        return False
