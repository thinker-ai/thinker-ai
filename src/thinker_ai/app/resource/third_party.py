from typing import Set

from thinker_ai.agent.tools.tool_data_type import Tool
from thinker_ai.common.resource import Resource


class ThirdParty(Resource):
    base_url:str
    services:Set[Tool]