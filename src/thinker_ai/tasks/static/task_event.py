
from typing import Any

from thinker_ai.common.serializable import Serializable


class WorkEvent(Serializable):
    id: str
    source_id: str
    name: str
    payload: Any

