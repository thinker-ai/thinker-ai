import importlib
import uuid
from typing import TypeVar, Type, Optional, Any

T = TypeVar('T')


def get_class_from_full_class_name(full_class_name: str):
    if '.' in full_class_name:
        module_name, class_name = full_class_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        scenario_class = getattr(module, class_name)
    else:
        # 假设类名在当前命名空间中
        scenario_class = globals().get(full_class_name)
        if scenario_class is None:
            raise ValueError(f"Class '{full_class_name}' not found in the current namespace.")
    return scenario_class


def from_class_name(cls: Type[T], full_class_name: str, **kwargs: Any) -> T:
    scenario_class = get_class_from_full_class_name(full_class_name)
    instance = scenario_class(**kwargs)
    if not isinstance(instance, cls):
        raise TypeError(f"Class {full_class_name} is not a subclass of {cls.__name__}")
    return instance


class Event:
    def __init__(self, name: str, id: Optional[str] = str(uuid.uuid4()), publisher_id: Optional[str] = None,
                 payload: Optional[Any] = None):
        self.id = id
        self.name = name
        self.publisher_id = publisher_id
        self.payload = payload


class Command:
    def __init__(self, name: str, target: str, id: Optional[str] = str(uuid.uuid4()), payload: Optional[Any] = None):
        self.id = id
        self.name = name
        self.target = target
        self.payload = payload


class ExecutorDescription:
    def __init__(self, data: dict):
        self.on_command = data["on_command"]
        if data.get("pre_check_list"):
            self.pre_check_list: list[str] = [item for item in data.get("pre_check_list")]
        self.full_class_name = data["full_class_name"]
        if data.get("post_check_list"):
            self.post_check_list: list[str] = [item for item in data.get("post_check_list")]
