import json
import uuid
from json import JSONDecodeError
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
)

from thinker_ai.common.exceptions import handle_exception
from thinker_ai.common.logs import logger
from thinker_ai.common.val_class import any_to_str_set, any_to_str, import_class

MESSAGE_ROUTE_FROM = "sent_from"
MESSAGE_ROUTE_TO = "send_to"
MESSAGE_ROUTE_CAUSE_BY = "cause_by"
MESSAGE_META_ROLE = "role"
MESSAGE_ROUTE_TO_ALL = "<all>"
MESSAGE_ROUTE_TO_NONE = "<none>"


class EnvMessage(BaseModel):
    """list[<role>: <content>]"""

    id: str = Field(default="", validate_default=True)  # According to Section 2.2.3.user_1.user_1 of RFC 135
    payload: Any
    role: str = "user"  # system / user / assistant
    cause_by: str = Field(default="", validate_default=True)
    sent_from: str = Field(default="", validate_default=True)
    send_to: set[str] = Field(default={MESSAGE_ROUTE_TO_ALL}, validate_default=True)

    @field_validator("id", mode="before")
    @classmethod
    def check_id(cls, id: str) -> str:
        return id if id else uuid.uuid4().hex

    @field_validator("cause_by", mode="before")
    @classmethod
    def check_cause_by(cls, cause_by: Any) -> str:
        return any_to_str(cause_by)

    @field_validator("sent_from", mode="before")
    @classmethod
    def check_sent_from(cls, sent_from: Any) -> str:
        return any_to_str(sent_from if sent_from else "")

    @field_validator("send_to", mode="before")
    @classmethod
    def check_send_to(cls, send_to: Any) -> set:
        return any_to_str_set(send_to if send_to else {MESSAGE_ROUTE_TO_ALL})

    @field_serializer("send_to", mode="plain")
    def ser_send_to(self, send_to: set) -> list:
        return list(send_to)

    def __init__(self, content: str = "", **data: Any):
        data["content"] = data.get("content", content)
        super().__init__(**data)

    def __setattr__(self, key, val):
        """Override `@property.setter`, convert non-string parameters into string parameters."""
        if key == MESSAGE_ROUTE_CAUSE_BY:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_FROM:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_TO:
            new_val = any_to_str_set(val)
        else:
            new_val = val
        super().__setattr__(key, new_val)

    def __str__(self):
        # prefix = '-'.join([self.role, str(self.cause_by)])
        if self.instruct_content:
            return f"{self.role}: {self.instruct_content.model_dump()}"
        return f"{self.role}: {self.content}"

    def __repr__(self):
        return self.__str__()

    def rag_key(self) -> str:
        """For search"""
        return self.content

    def to_dict(self) -> dict:
        """Return a dict containing `role` and `content` for the LLM call.l"""
        return {"role": self.role, "content": self.content}

    def dump(self) -> str:
        """Convert the object to q string"""
        return self.model_dump_json(exclude_none=True, warnings=False)

    @staticmethod
    @handle_exception(exception_type=JSONDecodeError, default_return=None)
    def load(val):
        """Convert the q string to object."""

        try:
            m = json.loads(val)
            id = m.get("id")
            if "id" in m:
                del m["id"]
            msg = EnvMessage(**m)
            if id:
                msg.id = id
            return msg
        except JSONDecodeError as err:
            logger.error(f"parse q failed: {val}, error:{err}")
        return None


def is_send_to(message: "EnvMessage", addresses: set):
    """Return whether it's consumer"""
    if MESSAGE_ROUTE_TO_ALL in message.send_to:
        return True

    for i in addresses:
        if i in message.send_to:
            return True
    return False
