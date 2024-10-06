import tensorflow as tf
import numpy as np
import os

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultWriteWeightCalculator, \
    DefaultTemporalLinkageUpdater, DefaultReadWeightCalculator, DefaultMemoryUpdater, DefaultUsageUpdater, \
    DefaultContentWeightCalculator
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess
from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import get_default_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义测试常量
BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
SEQUENCE_LENGTH = TIME_STEPS  # 保持一致性
INPUT_SIZE = 12  # 输入大小
EPSILON = 1e-6


class MemoryAccessSublayersFunctionalityTests(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessSublayersFunctionalityTests, self).setUp()
        # Initialize MemoryAccess module
        config = get_default_config(memory_size=MEMORY_SIZE, num_writes=NUM_WRITES, num_reads=NUM_READS, word_size=WORD_SIZE)
        self.module = MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES,
            epsilon=EPSILON,
            config=config
        )
        # Build the module to initialize weights
        # By calling the module once, Keras will automatically build sublayers
        batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)
        self.initial_state = self.module.get_initial_state(batch_size=batch_size_tensor)

        dummy_input = {
            'inputs': tf.zeros([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32),
            'prev_state': self.initial_state
        }
        _ = self.module(dummy_input, training=False)

    def testMemoryAccessCall(self):
        """Test the call method of MemoryAccess."""
        inputs = {
            'inputs': tf.random.normal([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]),
            'prev_state': self.initial_state
        }

        # Call the module
        outputs = self.module(inputs, training=False)

        # Extract read_words and final_state
        read_words = outputs['read_words']
        final_state = outputs['final_state']

        # Check shapes
        self.assertEqual(read_words.shape, [BATCH_SIZE, SEQUENCE_LENGTH, NUM_READS, WORD_SIZE])
        self.assertEqual(final_state.memory.shape, [BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        self.assertEqual(final_state.read_weights.shape, [BATCH_SIZE, SEQUENCE_LENGTH + self.initial_state.read_weights.shape[1], NUM_READS, MEMORY_SIZE])
        self.assertEqual(final_state.write_weights.shape, [BATCH_SIZE, SEQUENCE_LENGTH + self.initial_state.write_weights.shape[1], NUM_WRITES, MEMORY_SIZE])

    def testContentWeightCalculatorFunctionality(self):
        """Test the functionality of DefaultContentWeightCalculator."""
        content_weight_calculator = DefaultContentWeightCalculator( word_size=WORD_SIZE, epsilon=EPSILON)

        memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        keys = tf.random.normal([BATCH_SIZE, NUM_WRITES, WORD_SIZE])
        strengths = tf.random.uniform([BATCH_SIZE, NUM_WRITES], minval=0.1, maxval=10.0)

        content_weights = content_weight_calculator.compute(
            keys=keys,
            strengths=strengths,
            memory=memory
        )

        # Check output shape
        self.assertEqual(content_weights.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])

        # Check if weights sum to 1
        sum_weights = tf.reduce_sum(content_weights, axis=2)
        self.assertAllClose(sum_weights, tf.ones_like(sum_weights), atol=1e-3, msg="Content weights do not sum to 1")

    def testTemporalLinkageFunctionality(self):
        """Test the functionality of DefaultTemporalLinkageUpdater."""
        temporal_linkage = DefaultTemporalLinkageUpdater(memory_size=MEMORY_SIZE, num_writes=NUM_WRITES)

        write_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
        write_weights /= tf.reduce_sum(write_weights, axis=2, keepdims=True) + 1e-6

        prev_linkage = {
            'link': tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
            'precedence_weights': tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
        }

        updated_linkage = temporal_linkage.update_linkage(
            write_weights=write_weights,
            prev_linkage=prev_linkage,
            training=False
        )

        # Check output shapes
        self.assertEqual(updated_linkage['link'].shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE])
        self.assertEqual(updated_linkage['precedence_weights'].shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])

    def testUsageUpdaterFunctionality(self):
        """Test the functionality of DefaultUsageUpdater."""
        usage_updater = DefaultUsageUpdater(memory_size=MEMORY_SIZE, num_writes=NUM_WRITES, num_reads=NUM_READS)

        write_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
        free_gate = tf.random.uniform([BATCH_SIZE, NUM_READS], minval=0.0, maxval=1.0)
        read_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE], minval=0.0, maxval=1.0)
        prev_usage = tf.random.uniform([BATCH_SIZE, MEMORY_SIZE], minval=0.0, maxval=1.0)

        updated_usage = usage_updater.update_usage(
            write_weights=write_weights,
            free_gate=free_gate,
            read_weights=read_weights,
            prev_usage=prev_usage,
            training=False
        )

        # Check output shape
        self.assertEqual(updated_usage.shape, [BATCH_SIZE, MEMORY_SIZE])

        # Check if usage is in [0,1]
        self.assertTrue(tf.reduce_all(updated_usage >= 0.0))
        self.assertTrue(tf.reduce_all(updated_usage <= 1.0))

    def testWriteWeightCalculatorFunctionality(self):
        """Test the functionality of DefaultWriteWeightCalculator."""
        write_weight_calculator = DefaultWriteWeightCalculator(memory_size=MEMORY_SIZE, num_writes=NUM_WRITES)

        write_content_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
        allocation_gate = tf.random.uniform([BATCH_SIZE, NUM_WRITES], minval=0.0, maxval=1.0)
        write_gate = tf.random.uniform([BATCH_SIZE, NUM_WRITES], minval=0.0, maxval=1.0)
        prev_usage = tf.random.uniform([BATCH_SIZE, MEMORY_SIZE], minval=0.0, maxval=1.0)

        write_weights = write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # Check output shape
        self.assertEqual(write_weights.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])

        # Check if write_weights are non-negative
        self.assertTrue(tf.reduce_all(write_weights >= 0.0))

    def testReadWeightCalculatorFunctionality(self):
        """Test the functionality of DefaultReadWeightCalculator."""
        temporal_linkage = DefaultTemporalLinkageUpdater(memory_size=MEMORY_SIZE, num_writes=NUM_WRITES)
        read_weight_calculator = DefaultReadWeightCalculator(temporal_linkage=temporal_linkage, num_reads=NUM_READS, num_writes=NUM_WRITES)

        read_content_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE], minval=0.0, maxval=1.0)
        prev_read_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE], minval=0.0, maxval=1.0)
        link = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], minval=0.0, maxval=1.0)
        read_mode = tf.random.uniform([BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES], minval=0.0, maxval=1.0)

        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
            training=False
        )

        # Check output shape
        self.assertEqual(read_weights.shape, [BATCH_SIZE, NUM_READS, MEMORY_SIZE])

        # Check if read_weights are non-negative and sum to 1
        sum_weights = tf.reduce_sum(read_weights, axis=2)
        self.assertAllClose(sum_weights, tf.ones_like(sum_weights), atol=1e-3, msg="Read weights do not sum to 1")
        self.assertTrue(tf.reduce_all(read_weights >= 0.0))

    def testMemoryUpdaterFunctionality(self):
        """Test the functionality of DefaultMemoryUpdater."""
        memory_updater = DefaultMemoryUpdater()

        memory = tf.random.uniform([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], minval=-1.0, maxval=1.0)
        write_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
        erase_vectors = tf.random.uniform([BATCH_SIZE, NUM_WRITES, WORD_SIZE], minval=0.0, maxval=1.0)
        write_vectors = tf.random.uniform([BATCH_SIZE, NUM_WRITES, WORD_SIZE], minval=-1.0, maxval=1.0)

        memory_updated = memory_updater.update_memory(
            memory=memory,
            write_weights=write_weights,
            erase_vectors=erase_vectors,
            write_vectors=write_vectors
        )

        # Check output shape
        self.assertEqual(memory_updated.shape, [BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])

    def testSublayersRegistration(self):
        """Test whether all sublayers are correctly registered in the MemoryAccess module."""
        expected_sublayers = [
            'write_vectors_layer', 'erase_vectors_layer', 'write_gate_layer',
            'allocation_gate_layer', 'free_gate_layer', 'read_mode_layer',
            'write_keys_layer', 'write_strengths_layer', 'read_keys_layer',
            'read_strengths_layer'
        ]

        # Collect sublayers from self.module
        actual_sublayers = []
        for attr_name, attr_value in vars(self.module).items():
            if isinstance(attr_value, tf.keras.layers.Layer):
                actual_sublayers.append(attr_name)

        # Print actual sublayers for debugging
        print("Expected sublayers:", expected_sublayers)
        print("Actual sublayers:", actual_sublayers)

        # Check if all expected sublayers are in actual sublayers
        for sublayer in expected_sublayers:
            self.assertIn(sublayer, actual_sublayers, f"Sublayer '{sublayer}' is not registered.")


if __name__ == '__main__':
    tf.test.main()