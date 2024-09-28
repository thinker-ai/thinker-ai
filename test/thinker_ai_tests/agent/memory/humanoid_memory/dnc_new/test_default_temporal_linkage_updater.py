import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new.default_component import DefaultTemporalLinkageUpdater

def test_update_linkage():
    updater = DefaultTemporalLinkageUpdater(memory_size=128, num_writes=2, epsilon=1e-6)

    write_weights = tf.random.uniform([8, 2, 128])
    prev_linkage = {
        'link': tf.random.uniform([8, 2, 128, 128]),
        'precedence_weights': tf.random.uniform([8, 2, 128])
    }

    linkage_updated = updater.update_linkage(write_weights, prev_linkage, training=True)

    assert linkage_updated['link'].shape == (8, 2, 128, 128)
    assert linkage_updated['precedence_weights'].shape == (8, 2, 128)