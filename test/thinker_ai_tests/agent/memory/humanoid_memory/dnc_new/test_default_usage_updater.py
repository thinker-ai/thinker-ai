import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new.default_component import DefaultUsageUpdater

def test_update_usage():
    updater = DefaultUsageUpdater(memory_size=128, num_writes=2, num_reads=2, epsilon=1e-6)

    write_weights = tf.random.uniform([8, 2, 128])
    free_gate = tf.random.uniform([8, 2])
    read_weights = tf.random.uniform([8, 2, 128])
    prev_usage = tf.random.uniform([8, 128])

    usage_updated = updater.update_usage(write_weights, free_gate, read_weights, prev_usage, training=True)

    assert usage_updated.shape == (8, 128)
    assert tf.reduce_sum(usage_updated).numpy() > 0