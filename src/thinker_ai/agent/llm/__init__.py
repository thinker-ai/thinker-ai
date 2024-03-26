import os
from thinker_ai.agent.llm.claude2_api import Claude2, ClaudeConfig
from thinker_ai.agent.llm.gpt_api import GPT, GPT_Config
config = GPT_Config(api_key=os.environ.get("OPENAI_API_KEY"),
                    temperature=os.environ.get("temperature") or 0,
                    max_budget=os.environ.get("max_budget") or 3.0,
                    auto_max_tokens=False,
                    max_tokens_rsp=os.environ.get("max_tokens_rsp") or 2048,
                    proxy=os.environ.get("HTTP_PROXY") or None,
                    rpm=os.environ.get("rpm") or 10,
                    timeout=os.environ.get("timeout") or 3.0,
                    max_retries=os.environ.get("max_retries") or 3,
                    )

gpt = GPT(config)
# claude2 = Claude2(ClaudeConfig(api_key=os.environ.get("ANTHROPIC_API_KEY")))
