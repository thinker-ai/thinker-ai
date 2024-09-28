import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import WriteAllocation
from thinker_ai.agent.memory.humanoid_memory.dnc_new.default_component import DefaultWriteWeightCalculator

def test_compute_write_weights():
    write_allocation = WriteAllocation(memory_size=128, num_writes=2, epsilon=1e-6)
    calculator = DefaultWriteWeightCalculator(write_allocation=write_allocation)

    write_content_weights = tf.random.uniform([8, 2, 128])
    allocation_gate = tf.random.uniform([8, 2])
    write_gate = tf.random.uniform([8, 2])
    prev_usage = tf.random.uniform([8, 128])

    write_weights = calculator.compute_write_weights(write_content_weights, allocation_gate, write_gate, prev_usage, training=True)

    assert write_weights.shape == (8, 2, 128)
    assert tf.reduce_sum(write_weights).numpy() > 0