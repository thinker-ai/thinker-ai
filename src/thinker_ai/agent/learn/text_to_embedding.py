import thinker_ai.configs.config
from thinker_ai.agent.tools.openai_text_to_embedding import oas3_openai_text_to_embedding
from thinker_ai.configs.config import Config


async def text_to_embedding(text, model="text-embedding-ada-002", config: Config = thinker_ai.configs.config.config):
    """Text to embedding

    :param text: The text used for embedding.
    :param model: One of ['text-embedding-ada-002'], ID of the model to use. For more details, checkout: `https://api.openai.com/v1/models`.
    :param config: OpenAI config with API key, For more details, checkout: `https://platform.openai.com/account/api-keys`
    :return: A json object of :class:`ResultEmbedding` class if successful, otherwise `{}`.
    """
    openai_api_key = config.get_openai_llm().api_key
    proxy = config.get_openai_llm().proxy
    return await oas3_openai_text_to_embedding(text, model=model, openai_api_key=openai_api_key, proxy=proxy)
