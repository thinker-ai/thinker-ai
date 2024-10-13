import tensorflow as tf
import numpy as np
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


class SimpleController(tf.keras.layers.Layer):
    def __init__(self, output_size):
        super(SimpleController, self).__init__()
        self.dense = tf.keras.layers.Dense(output_size, activation='relu')

    def call(self, inputs, training=False):
        return self.dense(inputs)


class TestModel(tf.keras.Model):
    def __init__(self, controller_output_size, memory_access):
        super(TestModel, self).__init__()
        self.controller = SimpleController(controller_output_size)
        self.memory_access = memory_access

    def call(self, inputs, prev_state, training=False):
        controller_output = self.controller(inputs, training=training)
        memory_access_inputs = {
            'inputs': controller_output,
            'prev_state': prev_state
        }
        output = self.memory_access(memory_access_inputs, training=training)
        return output['read_words'], output['final_state']


class MemoryAccessGradientTest(tf.test.TestCase):
    def setUp(self):
        super(MemoryAccessGradientTest, self).setUp()
        tf.random.set_seed(42)
        np.random.seed(42)

        self.batch_size = 2
        self.input_size = 32
        self.controller_output_size = 64
        self.memory_size = 16
        self.word_size = 8
        self.num_reads = 1
        self.num_writes = 1

        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size
        )

        self.model = TestModel(self.controller_output_size, self.memory_access)

        self.controller_input = tf.random.uniform([self.batch_size, self.input_size])
        self.initial_state = self.memory_access.get_initial_state(self.batch_size)
        self.target = tf.random.uniform([self.batch_size, self.num_reads, self.word_size])

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

    def test_gradients_exist(self):
        with tf.GradientTape() as tape:
            read_words, final_state = self.model(self.controller_input, self.initial_state, training=False)
            loss = self.loss_fn(self.target, read_words)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        for var, grad in zip(trainable_vars, gradients):
            self.assertIsNotNone(grad, f"Gradient for variable {var.name} is None")
            tf.debugging.assert_all_finite(grad, f"Gradient for variable {var.name} has NaN or Inf values")


if __name__ == '__main__':
    tf.test.main()
