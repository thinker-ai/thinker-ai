import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new.default_component import DefaultMemoryUpdater

def test_update_memory():
    memory_updater = DefaultMemoryUpdater()

    memory = tf.random.uniform([8, 128, 32])
    write_weights = tf.random.uniform([8, 2, 128])
    erase_vectors = tf.random.uniform([8, 2, 32])
    write_vectors = tf.random.uniform([8, 2, 32])

    memory_updated = memory_updater.update_memory(memory, write_weights, erase_vectors, write_vectors)

    assert memory_updated.shape == (8, 128, 32)
    assert tf.reduce_sum(memory_updated).numpy() > 0