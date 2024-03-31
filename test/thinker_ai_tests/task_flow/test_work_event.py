import unittest

from thinker_ai.task_flow.task_event import WorkEvent


class TestWorkEvent(unittest.TestCase):
    def test_serialize_deserialize(self):
        # 创建一个包含各种类型payload的WorkEvent实例
        original_event = WorkEvent(
            id="1",
            source_id="source_1",
            name="EventName",
            payload={"key": "value", "number": 42}
        )

        # 序列化
        serialized = original_event.serialize()

        # 反序列化
        deserialized_event = WorkEvent.deserialize(serialized)

        # 检查反序列化得到的对象是否与原始对象相等
        self.assertEqual(original_event.id, deserialized_event.id)
        self.assertEqual(original_event.source_id, deserialized_event.source_id)
        self.assertEqual(original_event.name, deserialized_event.name)
        self.assertEqual(original_event.payload, deserialized_event.payload)


if __name__ == "__main__":
    unittest.main()
