from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess

import tensorflow as tf


class ShardManager:
    def __init__(self, num_shards, memory_args):
        """
        初始化分片管理器。

        Args:
            num_shards (int): 分片数量。
            memory_args (dict): MemoryAccess 初始化参数。
        """
        self.num_shards = num_shards
        self.memory_args = memory_args
        self.shards = [MemoryAccess(**memory_args) for _ in range(num_shards)]

    def _hash_users(self, user_ids):
        """
        通过哈希函数为批量用户分配分片ID。

        Args:
            user_ids (tf.Tensor): [batch_size]

        Returns:
            tf.Tensor: [batch_size], 分片ID
        """
        # 使用 TensorFlow 的哈希函数，将用户ID转换为分片ID
        hash_digest = tf.strings.to_hash_bucket_fast(tf.strings.as_string(user_ids), self.num_shards)
        return hash_digest  # [batch_size]

    def assign_users(self, user_ids):
        """
        为一批用户分配分片ID。

        Args:
            user_ids (tf.Tensor): [batch_size]

        Returns:
            tf.Tensor: [batch_size], 分片ID
        """
        return self._hash_users(user_ids)

    def get_shard(self, shard_id):
        """
        根据分片ID获取对应的分片实例。

        Args:
            shard_id (int): 分片ID

        Returns:
            MemoryAccess: 对应的 MemoryAccess 实例
        """
        return self.shards[shard_id]


class MemoryAccessWithUserEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_users, user_embedding_dim, shard_manager, name='memory_access_with_user_embedding'):
        """
        带有用户嵌入和分片管理的 MemoryAccess 模块。

        Args:
            num_users (int): 用户总数。
            user_embedding_dim (int): 用户嵌入维度。
            shard_manager (ShardManager): 分片管理器实例。
            name (str): 层的名称。
        """
        super(MemoryAccessWithUserEmbedding, self).__init__(name=name)
        self.num_users = num_users
        self.user_embedding_dim = user_embedding_dim
        self.shard_manager = shard_manager

        # 用户嵌入层
        self.user_embeddings = tf.keras.layers.Embedding(
            input_dim=num_users,
            output_dim=user_embedding_dim,
            name='user_embeddings'
        )

        # 处理用户嵌入与输入的融合层
        self.embedding_processor = tf.keras.layers.Dense(
            units=128,  # 可以根据需要调整
            activation=None,
            name='embedding_processor'
        )

    def call(self, inputs, training=False):
        """
        前向传播方法。

        Args:
            inputs (dict): 包含 'inputs' 和 'user_id'
                - 'inputs': [batch_size, sequence_length, input_size]
                - 'user_id': [batch_size]
            training (bool): 是否在训练模式

        Returns:
            dict: 包含 'read_words' 和 'final_state'
        """
        controller_input = inputs['inputs']  # [batch_size, sequence_length, input_size]
        user_id = inputs['user_id']  # [batch_size]

        # 获取用户嵌入
        user_embeds = self.user_embeddings(user_id)  # [batch_size, user_embedding_dim]

        # 处理用户嵌入
        processed_embeds = self.embedding_processor(user_embeds)  # [batch_size, processed_dim]

        # 将嵌入向量扩展到时间步维度并与输入结合
        processed_embeds_expanded = tf.expand_dims(processed_embeds, axis=1)  # [batch_size, 1, processed_dim]
        combined_inputs = tf.concat([controller_input, processed_embeds_expanded],
                                    axis=-1)  # [batch_size, sequence_length, input_size + processed_dim]

        # 分配用户到分片
        shard_ids = self.shard_manager.assign_users(user_id)  # [batch_size]

        # 初始化输出列表
        read_words_all = []
        final_states = []

        # 使用 tf.vectorized_map 进行并行处理
        def process_sample(args):
            user_input, shard_id = args
            shard = self.shard_manager.get_shard(shard_id)
            # 获取初始状态
            prev_state = shard.get_initial_state(batch_shape=[1])
            # 运行 MemoryAccess 模块
            output = shard({'inputs': user_input, 'prev_state': prev_state}, training=training)
            return output['read_words'], output['final_state']

        # 使用 tf.vectorized_map 处理每个样本
        read_words, final_states = tf.vectorized_map(
            process_sample,
            (combined_inputs, shard_ids),
            parallel_iterations=32,
            back_prop=training
        )

        # 合并输出
        read_words = tf.reshape(read_words, [tf.shape(controller_input)[0], tf.shape(controller_input)[1],
                                             self.shard_manager.shards[0].num_reads,
                                             self.shard_manager.shards[0].word_size])
        # 合并 final_states （假设只保留最终状态，不进行进一步处理）
        final_state = final_states[-1]

        return {'read_words': read_words, 'final_state': final_state}

    def get_initial_state(self, batch_shape, initial_time_steps=0):
        """
        返回 MemoryAccessWithUserEmbedding 模块的初始状态。

        Args:
            batch_shape (tuple | list): 批次形状，例如 [batch_size]
            initial_time_steps (int, optional): 初始时间步数。默认值为 0。

        Returns:
            list of AccessState: 每个分片的初始状态列表。
        """
        initial_states = []
        for shard in self.shard_manager.shards:
            initial_state = shard.get_initial_state(batch_shape=batch_shape, initial_time_steps=initial_time_steps)
            initial_states.append(initial_state)
        return initial_states

    def state_size(self):
        """
        返回 MemoryAccessWithUserEmbedding 模块的状态大小。

        Returns:
            list of AccessState: 每个分片的状态大小列表。
        """
        return [shard.state_size() for shard in self.shard_manager.shards]
