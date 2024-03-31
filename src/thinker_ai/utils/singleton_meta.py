from threading import Lock


class SingletonMeta(type):
    """一个用于创建单例的元类，保证每个类只有一个实例，并提供全局访问点。"""
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]