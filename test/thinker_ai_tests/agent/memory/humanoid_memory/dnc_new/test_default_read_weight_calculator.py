import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import TemporalLinkage
from thinker_ai.agent.memory.humanoid_memory.dnc_new.default_component import DefaultReadWeightCalculator

def test_compute_read_weights():
    temporal_linkage = TemporalLinkage(memory_size=128, num_writes=2, epsilon=1e-6)
    calculator = DefaultReadWeightCalculator(temporal_linkage=temporal_linkage)

    read_content_weights = tf.random.uniform([8, 2, 128])
    prev_read_weights = tf.random.uniform([8, 2, 128])
    link = tf.random.uniform([8, 2, 128, 128])
    read_mode = tf.random.uniform([8, 2, 1 + 2 * 2])

    read_weights = calculator.compute_read_weights(read_content_weights, prev_read_weights, link, read_mode, training=True)

    assert read_weights.shape == (8, 2, 128)
    assert tf.reduce_sum(read_weights).numpy() > 0