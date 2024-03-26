
from typing import Any

from thinker_ai.core.serializable import Serializable


class WorkEvent(Serializable):
    id: str
    source_id: str
    name: str
    payload: Any

