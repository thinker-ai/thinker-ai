import tensorflow as tf
import numpy as np

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultTemporalLinkageUpdater


class TemporalLinkageTest(tf.test.TestCase):
    def setUp(self):
        super(TemporalLinkageTest, self).setUp()
        self.memory_size = 3
        self.num_writes = 2
        self.temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=self.memory_size,
            num_writes=self.num_writes
        )

    def test_precedence_weights_update(self):
        """
        Test that precedence weights are updated correctly without epsilon.
        """
        batch_size = 1
        write_weights = tf.constant([[[0.1, 0.2, 0.3],
                                      [0.4, 0.1, 0.2]]], dtype=tf.float32)  # [1, 2, 3]
        prev_precedence_weights = tf.constant([[[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 2, 3]
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': prev_precedence_weights
        }

        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        write_sum = tf.reduce_sum(write_weights, axis=2, keepdims=True)
        expected_precedence_weights = (1 - write_sum) * prev_precedence_weights + write_weights

        self.assertAllClose(updated_linkage['precedence_weights'], expected_precedence_weights, atol=1e-6)

    def test_link_matrix_update(self):
        """
        Test that the link matrices are updated correctly.
        """
        batch_size = 1
        write_weights = tf.constant([[[0.2, 0.5, 0.3],
                                      [0.1, 0.6, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_precedence_weights = tf.constant([[[0.1, 0.2, 0.7],
                                                [0.3, 0.4, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        prev_linkage = {
            'link': prev_link,
            'precedence_weights': prev_precedence_weights
        }

        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        # Manually compute expected link
        write_sum = tf.reduce_sum(write_weights, axis=2, keepdims=True)
        updated_precedence_weights = (1 - write_sum) * prev_precedence_weights + write_weights

        write_weights_i = tf.expand_dims(write_weights, axis=3)  # [batch_size, num_writes, memory_size, 1]
        precedence_weights_j = tf.expand_dims(updated_precedence_weights, axis=2)  # [batch_size, num_writes, 1, memory_size]

        new_link = write_weights_i * precedence_weights_j  # [batch_size, num_writes, memory_size, memory_size]

        # Remove self-links
        identity = tf.eye(self.memory_size, batch_shape=[self.num_writes], dtype=tf.float32)
        identity = tf.expand_dims(identity, axis=0)
        new_link = new_link * (1 - identity)

        expected_link = (1 - write_weights_i - tf.expand_dims(write_weights, axis=2)) * prev_link + new_link
        expected_link = tf.clip_by_value(expected_link, 0.0, 1.0)

        self.assertAllClose(updated_linkage['link'], expected_link, atol=1e-6)

    def test_directional_read_weights(self):
        """
        Test the computation of directional read weights.
        """
        batch_size = 1
        num_reads = 1

        link = tf.constant([[
            [[0.0, 0.1, 0.2],
             [0.3, 0.0, 0.4],
             [0.5, 0.6, 0.0]],
            [[0.0, 0.2, 0.1],
             [0.1, 0.0, 0.5],
             [0.4, 0.2, 0.0]]
        ]], dtype=tf.float32)  # [1, 2, 3, 3]

        prev_read_weights = tf.constant([[[0.2, 0.5, 0.3]]], dtype=tf.float32)  # [1, 1, 3]

        # Compute forward directional weights
        directional_weights = self.temporal_linkage.directional_read_weights(
            link, prev_read_weights, forward=True
        )

        # Manually compute expected directional weights
        expected_directional_weights = []
        for i in range(self.num_writes):
            link_i = link[:, i, :, :]  # [1, 3, 3]
            prev_read_weights_reshaped = tf.expand_dims(prev_read_weights, axis=2)  # [1, 1, 1, 3]
            link_i_expanded = tf.expand_dims(link_i, axis=1)  # [1, 1, 3, 3]
            weight = tf.matmul(prev_read_weights_reshaped, link_i_expanded)  # [1, 1, 1, 3]
            weight = tf.squeeze(weight, axis=2)  # [1, 1, 3]
            expected_directional_weights.append(weight)

        expected_directional_weights = tf.stack(expected_directional_weights, axis=2)  # [1, 1, 2, 3]

        self.assertAllClose(directional_weights, expected_directional_weights, atol=1e-6)

    def test_state_size(self):
        """
        Test the state_size method.
        """
        expected_state_size = {
            'link': tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.num_writes, self.memory_size])
        }
        self.assertEqual(self.temporal_linkage.state_size(), expected_state_size)

    def test_no_self_links(self):
        """
        Test that the link matrices have zeros on the diagonal (no self-links).
        """
        batch_size = 1
        write_weights = tf.constant([[[0.3, 0.6, 0.1],
                                      [0.2, 0.5, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }
        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        link_matrices = updated_linkage['link'].numpy()
        for i in range(self.num_writes):
            # Extract the link matrix for write head i
            link_matrix = link_matrices[0, i, :, :]
            # Check that the diagonal is zero
            diagonal = np.diag(link_matrix)
            self.assertAllClose(diagonal, np.zeros(self.memory_size), atol=1e-6)

    def test_gradient_flow(self):
        """
        Test that gradients flow correctly through the update_linkage method.
        """
        batch_size = 1
        write_weights = tf.Variable([[[0.5, 0.3, 0.2],
                                      [0.1, 0.6, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }

        with tf.GradientTape() as tape:
            updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)
            loss = tf.reduce_sum(updated_linkage['precedence_weights'])

        grads = tape.gradient(loss, [write_weights])
        self.assertIsNotNone(grads[0])
        self.assertGreater(tf.norm(grads[0]).numpy(), 1e-6)

if __name__ == '__main__':
    tf.test.main()