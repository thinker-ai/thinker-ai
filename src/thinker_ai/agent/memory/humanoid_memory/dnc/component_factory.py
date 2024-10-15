# component_factory.py
from typing import Optional, Dict, Any
import importlib
import inspect
class ComponentFactory:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        memory_size: int = None,
        word_size: int = None,
        num_reads: int = None,
        num_writes: int = None
    ):
        self.config = config or {}
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = num_writes

    def create_all_components(self) -> Dict[str, Any]:
        content_weight_calculator = self.create_content_weight_calculator()
        write_weight_calculator = self.create_write_weight_calculator()
        temporal_linkage_updater = self.create_temporal_linkage_updater()
        read_weight_calculator = self.create_read_weight_calculator()
        usage_updater = self.create_usage_updater()
        memory_updater = self.create_memory_updater()

        return {
            'content_weight_calculator': content_weight_calculator,
            'write_weight_calculator': write_weight_calculator,
            'temporal_linkage_updater': temporal_linkage_updater,
            'read_weight_calculator': read_weight_calculator,
            'usage_updater': usage_updater,
            'memory_updater': memory_updater
        }

    def create_content_weight_calculator(self):
        config = self.config.get('ContentWeightCalculator', {})
        print(f"ContentWeightCalculator config: {config}")  # 添加打印
        class_path = config.pop('class_path', 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultContentWeightCalculator')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_write_weight_calculator(self):
        config = self.config.get('WriteWeightCalculator', {})
        class_path = config.pop('class_path', 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultWriteWeightCalculator')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_temporal_linkage_updater(self):
        config = self.config.get('TemporalLinkageUpdater', {})
        class_path = config.pop('class_path', 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultTemporalLinkageUpdater')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_read_weight_calculator(self):
        config = self.config.get('ReadWeightCalculator', {})
        class_path = config.pop('class_path', 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultReadWeightCalculator')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_usage_updater(self):
        config = self.config.get('UsageUpdater', {})
        class_path = config.pop('class_path', 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultUsageUpdater')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def create_memory_updater(self):
        config = self.config.get('MemoryUpdater', {})
        class_path = config.pop('class_path', 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultMemoryUpdater')
        cls = self._load_class(class_path)
        filtered_config = self._filter_kwargs(cls, config)
        return cls(**filtered_config)

    def _load_class(self, class_path: str):
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls

    def _filter_kwargs(self, func, kwargs):
        """
        过滤 kwargs，只保留 func 的 __init__ 方法中接受的参数。
        """
        try:
            # 获取 func 的 __init__ 方法
            init_method = func.__init__

            # 检查 __init__ 是否来自于 object，如果是，则没有参数
            if init_method == object.__init__:
                return {}

            # 使用 inspect 获取参数列表
            signature = inspect.signature(init_method)
            func_params = signature.parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
            return filtered_kwargs
        except (AttributeError, ValueError, TypeError):
            # 如果无法获取 __init__ 方法的签名，返回空的 kwargs
            return {}