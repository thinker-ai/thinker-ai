# component_factory.py
import importlib
from typing import Dict, Any
import inspect

from thinker_ai.agent.memory.humanoid_memory.dnc.component_interface import (
    WriteWeightCalculator,
    ReadWeightCalculator,
    ContentWeightCalculator,
    UsageUpdater,
    TemporalLinkageUpdater,
    MemoryUpdater
)


class ComponentFactory:
    """
    工厂类，用于根据配置动态加载和实例化组件实现。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化工厂，保存配置。

        Args:
            config (Dict[str, Any]): 配置字典，指定各组件的实现类及其参数。
        """
        self.config = config

    def _load_class(self, class_path: str):
        """
        动态加载指定路径的类。

        Args:
            class_path (str): 类的完整路径，例如 'module.submodule.ClassName'

        Returns:
            type: 加载的类
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls

    def _filter_kwargs(self, cls, config):
        """
        过滤配置字典，只保留构造函数接受的参数。

        Args:
            cls (type): 类对象
            config (Dict[str, Any]): 配置字典

        Returns:
            Dict[str, Any]: 过滤后的配置字典
        """
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        valid_params = {
            k: v for k, v in config.items()
            if k in params and k != 'self'
        }
        return valid_params

    def create_write_weight_calculator(self) -> WriteWeightCalculator:
        config = self.config.get('WriteWeightCalculator', {})
        class_path = config.pop('class_path',
                                'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultWriteWeightCalculator')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_temporal_linkage_updater(self) -> TemporalLinkageUpdater:
        config = self.config.get('TemporalLinkageUpdater', {})
        class_path = config.pop('class_path',
                                'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultTemporalLinkageUpdater')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_read_weight_calculator(self, temporal_linkage: TemporalLinkageUpdater) -> ReadWeightCalculator:
        config = self.config.get('ReadWeightCalculator', {})
        class_path = config.pop('class_path',
                                'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultReadWeightCalculator')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(temporal_linkage=temporal_linkage, **filtered_config)

    def create_content_weight_calculator(self) -> ContentWeightCalculator:
        config = self.config.get('ContentWeightCalculator', {})
        class_path = config.pop('class_path',
                                'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultContentWeightCalculator')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_usage_updater(self) -> UsageUpdater:
        config = self.config.get('UsageUpdater', {})
        class_path = config.pop('class_path',
                                'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultUsageUpdater')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_memory_updater(self) -> MemoryUpdater:
        config = self.config.get('MemoryUpdater', {})
        class_path = config.pop('class_path',
                                'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultMemoryUpdater')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_all_components(self):
        """
        创建所有组件，确保依赖关系正确处理。
        """
        write_weight_calculator = self.create_write_weight_calculator()
        temporal_linkage_updater = self.create_temporal_linkage_updater()
        read_weight_calculator = self.create_read_weight_calculator(temporal_linkage=temporal_linkage_updater)
        content_weight_calculator = self.create_content_weight_calculator()
        usage_updater = self.create_usage_updater()
        memory_updater = self.create_memory_updater()

        return {
            'write_weight_calculator': write_weight_calculator,
            'read_weight_calculator': read_weight_calculator,
            'content_weight_calculator': content_weight_calculator,
            'usage_updater': usage_updater,
            'temporal_linkage_updater': temporal_linkage_updater,
            'memory_updater': memory_updater
        }