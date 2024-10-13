import os
import pickle


class CacheManager:
    def __init__(self, cache_dir: str = "./cache", max_cache_size: int = 1024):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size  # 最大缓存大小（可以根据需求调整）
        self.memory_cache = {}  # 用于存储缓存数据
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """确保缓存目录存在。"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def read_from_cache(self, cache_key: str):
        """从缓存中读取数据，如果缓存不存在，则返回 None。"""
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # 如果缓存数据不在内存中，则从持久化存储中读取
        cache_file_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)

        return None

    def write_to_cache(self, cache_key: str, data):
        """将数据写入缓存，并持久化到磁盘。"""
        # 首先将数据写入内存缓存
        self.memory_cache[cache_key] = data

        # 如果缓存大小超过限制，则持久化
        if len(self.memory_cache) > self.max_cache_size:
            self._persist_to_disk(cache_key, data)

    def _persist_to_disk(self, cache_key: str, data):
        """将数据持久化到磁盘。"""
        cache_file_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_file_path, 'wb') as f:
            pickle.dump(data, f)

        # 清理内存中的数据（只保留部分数据）
        if len(self.memory_cache) > self.max_cache_size:
            del self.memory_cache[cache_key]

    def restore_from_cache(self, cache_key: str):
        """从缓存或持久化存储中恢复数据。"""
        return self.read_from_cache(cache_key)

    def clear_cache(self):
        """清理缓存目录和内存缓存。"""
        self.memory_cache.clear()
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            os.remove(file_path)
