import base64

import thinker_ai.configs.config
from thinker_ai.agent.provider.llm import LLM
from thinker_ai.agent.tools.text_to_image import oas3_thinker_ai_text_to_image
from thinker_ai.agent.tools.openai_text_to_image import oas3_openai_text_to_image
from thinker_ai.configs.config import Config
from thinker_ai.configs.const import BASE64_FORMAT
from thinker_ai.utils.s3 import S3


async def text_to_image(text, size_type: str = "512x512", config: Config = thinker_ai.configs.config.config):
    """Text to image

    :param text: The text used for image conversion.
    :param size_type: If using OPENAI, the available size options are ['256x256', '512x512', '1024x1024'], while for ThinkerAI, the options are ['512x512', '512x768'].
    :param config: Config
    :return: The image data is returned in Base64 encoding.
    """
    image_declaration = "data:image/png;base64,"

    model_url = config.thinker_ai_tti_url
    if model_url:
        binary_data = await oas3_thinker_ai_text_to_image(text, size_type, model_url)
    elif config.get_openai_llm():
        llm = LLM(llm_config=config.get_openai_llm())
        binary_data = await oas3_openai_text_to_image(text, size_type, llm=llm)
    else:
        raise ValueError("Missing necessary parameters.")
    base64_data = base64.b64encode(binary_data).decode("utf-8")

    s3 = S3(config.s3)
    url = await s3.cache(data=base64_data, file_ext=".png", format=BASE64_FORMAT)
    if url:
        return f"![{text}]({url})"
    return image_declaration + base64_data if base64_data else ""
