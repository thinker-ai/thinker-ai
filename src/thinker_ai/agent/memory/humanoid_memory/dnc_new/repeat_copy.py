import collections
import numpy as np
import tensorflow as tf

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations', 'target', 'mask'))


def masked_sigmoid_cross_entropy(logits, target, mask, time_average=False, log_prob_in_bits=False):
    """添加计算目标序列负对数似然 (NLL) 的操作。

    Args:
      logits: `Tensor`，用于通过 sigmoid(`logits`) 生成伯努利分布参数的激活值。
      target: 时间为主的 `Tensor`，表示目标序列。
      mask: 时间为主的 `Tensor`，逐元素与损失值相乘，用于屏蔽不相关的时间步长。
      time_average: 可选项，若为 True，则对时间维度进行平均（默认对时间步长进行求和）。
      log_prob_in_bits: 若为 True，则以比特为单位表示对数概率（默认以 nats 表示）。

    Returns:
      一个表示目标序列负对数似然 (NLL) 的 `Tensor`。
    """
    # Compute sigmoid cross entropy with logits
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)

    # Sum across the last dimension (binary vector) for each timestep and batch
    loss_time_batch = tf.reduce_sum(xent, axis=2)

    # Apply mask and sum across time
    loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)

    # Get batch size and normalize the loss
    batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

    if time_average:
        mask_count = tf.reduce_sum(mask, axis=0)
        loss_batch /= (mask_count + np.finfo(np.float32).eps)

    loss = tf.reduce_sum(loss_batch) / batch_size

    # Convert to bits if specified
    if log_prob_in_bits:
        loss /= tf.math.log(2.)

    return loss


def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
    """生成数据中序列的可视化字符串。

    Args:
      data: 需要可视化的数据。
      batch_size: 批次大小，即数据中包含的序列数量。
      model_output: 可选的模型输出张量，用于与数据一起可视化。
      whole_batch: 是否可视化整个批次。如果为 False，则仅可视化第一个样本。

    Returns:
      一个字符串，表示数据批次的可视化。
    """

    def _readable(datum):
        return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

    obs_batch = data.observations
    targ_batch = data.target

    iterate_over = range(batch_size) if whole_batch else range(1)

    batch_strings = []
    for batch_index in iterate_over:
        obs = obs_batch[:, batch_index, :]
        targ = targ_batch[:, batch_index, :]

        obs_channels = range(obs.shape[1])
        targ_channels = range(targ.shape[1])
        obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
        targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]

        readable_obs = 'Observations:\n' + '\n'.join(obs_channel_strings)
        readable_targ = 'Targets:\n' + '\n'.join(targ_channel_strings)
        strings = [readable_obs, readable_targ]

        if model_output is not None:
            output = model_output[:, batch_index, :]
            output_strings = [_readable(output[:, i]) for i in targ_channels]
            strings.append('Model Output:\n' + '\n'.join(output_strings))

        batch_strings.append('\n\n'.join(strings))

    return '\n' + '\n\n\n\n'.join(batch_strings)


class RepeatCopy(tf.keras.layers.Layer):
    """RepeatCopy 是一个用于生成重复随机二进制模式任务的序列数据生成器。

当调用这个类的实例时，它将返回一个包含观察序列、目标序列和掩码的 DatasetTensors 元组。
这些序列中的每一个张量的前两个维度分别代表序列的位置和批次索引。掩码张量中的值 `mask[t, b]`
等于1，当且仅当 `targ[t, b, :]` 应该受到惩罚时，也就是说，掩码值为1的位置表示对应的目标值应被计算损失。

对于每次生成的数据，观察序列由独立同分布的随机二进制向量（以及一些标记位）组成。

目标序列由这个二进制模式重复一定次数的结果（以及一些标记位）组成。为了更好地说明，
以下是一个批次中的单个元素的示意图：
```none
  时间轴 ------------------------------------------>

                +-------------------------------+
掩码:           |0000000001111111111111111111111|
                +-------------------------------+

                +-------------------------------+
目标序列:        |                              1| '结束标记' 通道。
                |         101100110110011011001 |
                |         010101001010100101010 |
                +-------------------------------+

                +-------------------------------+
观察序列:        | 1011001                       |
                | 0101010                       |
                |1                              | '开始标记' 通道
                |        3                      | '重复次数' 通道。
                +-------------------------------+
  ```
在上述示意图中：
    1.	掩码序列决定了模型需要在哪些时间步进行损失计算。在0的时间步中，模型的输出不会被计算损失；而在1的时间步中，模型的输出将参与损失计算。
    2.	目标序列代表要学习的正确输出，它包含一个以 1011001 和 0101010 组成的二进制序列，并被标记为需要在最后一个时间步（时间步25）进行处理。
    3.	观察序列是模型接收到的输入序列，包含二进制随机向量以及一些额外的标记。第一行中标记了“开始标记”，表示序列的开始。接下来是该随机序列的重复次数，随后是随机的二进制向量本身。
生成的序列详细说明

    •	随机模式的长度和重复次数都是根据统一分布生成的离散随机变量。这些参数可以在类的构造时配置。
    •	观察序列有两个额外的通道，分别用于标记开始和结束。第一个通道在第一个时间步标记为1，其余时间步为0；另一个通道在二进制模式结束后标记了该模式的重复次数。这个重复次数被标准化为0到1之间的浮点数，以便神经网络更容易地学习并表示这个重复次数。
    •	为了使网络能够在重复次数更多的实例上顺利进行评估，用户可以配置重复次数的范围，用于标准化编码。
    •	目标序列从观察序列结束后开始，两个序列通过在序列末尾补0来对齐，确保批次内所有序列的长度一致。
    •	额外的填充保证了批次中的所有序列具有相同的形状。也就是说，每个批次中的所有序列在计算图中都拥有相同的维度。

样本数据生成流程

    1.	随机生成一个长度为 seq_length 的二进制序列。
    2.	随机选择重复次数 repeat_count，将该序列重复相应的次数。
    3.	为观察序列添加特殊标记（例如开始标记和重复次数标记）。
    4.	生成的掩码标记出需要进行损失计算的时间步，掩码值为1的时间步表示需要损失计算，值为0的时间步则被忽略。
"""

    def __init__(self, num_bits=6, batch_size=1, min_length=1, max_length=1, min_repeats=1, max_repeats=2, norm_max=10,
                 log_prob_in_bits=False, time_average_cost=False, name='repeat_copy'):
        """
        Args:
          name: 用于生成器实例的名称（用于命名作用域 purposes）。
          num_bits: 每个随机二进制向量的维度。
          batch_size: 每次生成的批次大小（即每个批次包含的序列数）。
          min_length: 观测模式中随机二进制向量的最小数量（序列的最短长度）。
          max_length: 观测模式中随机二进制向量的最大数量（序列的最长长度）。
          min_repeats: 观测模式在目标序列中重复的最小次数。
          max_repeats: 观测模式在目标序列中重复的最大次数。
          norm_max: 用于对观测序列中重复次数的编码进行归一化的上限值。
          log_prob_in_bits: 默认情况下，以 nats 表示对数概率。如果为 True，则以比特表示对数概率。
          time_average_cost: 如果为 True，在计算损失时会对时间维度进行平均。否则，对时间维度进行求和。
        """
        super(RepeatCopy, self).__init__(name=name)
        self._batch_size = batch_size
        self._num_bits = num_bits
        self._min_length = min_length
        self._max_length = max_length
        self._min_repeats = min_repeats
        self._max_repeats = max_repeats
        self._norm_max = norm_max
        self._log_prob_in_bits = log_prob_in_bits
        self._time_average_cost = time_average_cost

    def _normalise(self, val):
        return val / self._norm_max

    def _unnormalise(self, val):
        return val * self._norm_max

    def call(self, inputs: tf.Tensor) -> DatasetTensors:
        """生成重复拷贝任务数据的核心逻辑。

        当模型调用 `call` 方法时，它会执行生成重复拷贝任务的数据逻辑。具体来说，
        `call` 方法接收输入张量，并生成用于训练模型的观察值、目标值和掩码。这个方法
        通常是通过 `__call__` 接口被调用的，并且被 Keras 框架用来定义前向传播。

        机制解释:
        1. `inputs`: TensorFlow 的张量对象，作为输入数据。它通常是从 `Layer` 类继承而来的。
           `call` 方法的输入张量 `inputs` 通常由模型的前一层生成，或者是训练时的输入数据。
        2. `call` 方法返回包含三个关键部分的 `DatasetTensors`：`observations`、`target` 和 `mask`。
           - `observations` 表示输入序列的数据。
           - `target` 是模型的目标输出，表示模型需要生成的重复序列。
           - `mask` 是二值掩码，用于在计算损失时确定哪些序列位置需要被计算和考虑。

        具体操作步骤:
        1. 通过调用 `tf.random.uniform` 随机生成每个批次样本的序列长度和重复次数。
        2. 生成用于观测的二进制序列，使用随机数生成器生成独立同分布的二进制序列作为输入。
        3. 扩展序列以包含两个额外的通道：一个起始标记位，另一个是重复次数标记位。
        4. 生成目标序列，该序列是输入序列的多次重复，并在最后添加结束标记。
        5. 生成掩码，该掩码指定目标序列的哪些部分在损失函数中被计算（即有效的预测区域）。
        6. 使用 TensorFlow 操作来填充不同样本长度的数据，使得每个批次中的样本序列长度一致。
        7. 最后，返回 `DatasetTensors`，包含 `observations`（观察值）、`target`（目标）和 `mask`（掩码）。

        Args:
          inputs: `Tensor`，作为输入序列的张量。

        Returns:
          包含三个部分的 `DatasetTensors`：
            - `observations`: 输入序列的张量，包含标记位。
            - `target`: 目标输出序列的张量，表示重复的序列。
            - `mask`: 掩码，指示哪些序列位置在损失计算中是有效的。
        """
        min_length, max_length = self._min_length, self._max_length
        min_reps, max_reps = self._min_repeats, self._max_repeats
        num_bits = self._num_bits
        batch_size = self._batch_size

        # We reserve one dimension for the num-repeats and one for the start-marker.
        full_obs_size = num_bits + 2
        full_targ_size = num_bits + 1

        # Samples each batch index's sequence length and the number of repeats.
        sub_seq_length_batch = tf.random.uniform([batch_size], minval=min_length, maxval=max_length + 1, dtype=tf.int32)
        num_repeats_batch = tf.random.uniform([batch_size], minval=min_reps, maxval=max_reps + 1, dtype=tf.int32)

        # Pads all the batches to have the same total sequence length.
        total_length_batch = sub_seq_length_batch * (num_repeats_batch + 1) + 3
        max_length_batch = tf.reduce_max(total_length_batch)
        residual_length_batch = max_length_batch - total_length_batch

        obs_batch_shape = [max_length_batch, batch_size, full_obs_size]
        targ_batch_shape = [max_length_batch, batch_size, full_targ_size]
        mask_batch_trans_shape = [batch_size, max_length_batch]

        obs_tensors = []
        targ_tensors = []
        mask_tensors = []

        for batch_index in range(batch_size):
            sub_seq_len = sub_seq_length_batch[batch_index]
            num_reps = num_repeats_batch[batch_index]

            # The observation pattern is a sequence of random binary vectors.
            obs_pattern_shape = [sub_seq_len, num_bits]
            obs_pattern = tf.cast(tf.random.uniform(obs_pattern_shape, minval=0, maxval=2, dtype=tf.int32), tf.float32)

            # The target pattern is the observation pattern repeated n times.
            targ_pattern_shape = [sub_seq_len * num_reps, num_bits]
            flat_obs_pattern = tf.reshape(obs_pattern, [-1])
            flat_targ_pattern = tf.tile(flat_obs_pattern, tf.stack([num_reps]))
            targ_pattern = tf.reshape(flat_targ_pattern, targ_pattern_shape)

            # Expand the obs_pattern to have two extra channels for flags.
            obs_flag_channel_pad = tf.zeros([sub_seq_len, 2])
            obs_start_flag = tf.one_hot([full_obs_size - 2], full_obs_size, on_value=1., off_value=0.)
            num_reps_flag = tf.one_hot([full_obs_size - 1], full_obs_size,
                                       on_value=self._normalise(tf.cast(num_reps, tf.float32)), off_value=0.)

            # Concatenate the flags with the observation pattern.
            obs = tf.concat([obs_pattern, obs_flag_channel_pad], 1)
            obs = tf.concat([obs_start_flag, obs], 0)
            obs = tf.concat([obs, num_reps_flag], 0)

            # Now do the same for the targ_pattern (it only has one extra channel).
            targ_flag_channel_pad = tf.zeros([sub_seq_len * num_reps, 1])
            targ_end_flag = tf.one_hot([full_obs_size - 2], full_targ_size, on_value=1., off_value=0.)
            targ = tf.concat([targ_pattern, targ_flag_channel_pad], 1)
            targ = tf.concat([targ, targ_end_flag], 0)

            obs_end_pad = tf.zeros([sub_seq_len * num_reps + 1, full_obs_size])
            targ_start_pad = tf.zeros([sub_seq_len + 2, full_targ_size])

            mask_off = tf.zeros([sub_seq_len + 2])
            mask_on = tf.ones([sub_seq_len * num_reps + 1])

            obs = tf.concat([obs, obs_end_pad], 0)
            targ = tf.concat([targ_start_pad, targ], 0)
            mask = tf.concat([mask_off, mask_on], 0)

            obs_tensors.append(obs)
            targ_tensors.append(targ)
            mask_tensors.append(mask)

        residual_obs_pad = [tf.zeros([residual_length_batch[i], full_obs_size]) for i in range(batch_size)]
        residual_targ_pad = [tf.zeros([residual_length_batch[i], full_targ_size]) for i in range(batch_size)]
        residual_mask_pad = [tf.zeros([residual_length_batch[i]]) for i in range(batch_size)]

        obs_tensors = [tf.concat([o, p], 0) for o, p in zip(obs_tensors, residual_obs_pad)]
        targ_tensors = [tf.concat([t, p], 0) for t, p in zip(targ_tensors, residual_targ_pad)]
        mask_tensors = [tf.concat([m, p], 0) for m, p in zip(mask_tensors, residual_mask_pad)]

        obs = tf.reshape(tf.concat(obs_tensors, 1), obs_batch_shape)
        targ = tf.reshape(tf.concat(targ_tensors, 1), targ_batch_shape)
        mask = tf.transpose(tf.reshape(tf.concat(mask_tensors, 0), mask_batch_trans_shape))

        return DatasetTensors(obs, targ, mask)

    def cost(self, logits, targ, mask):
        return masked_sigmoid_cross_entropy(
            logits,
            targ,
            mask,
            time_average=self._time_average_cost,
            log_prob_in_bits=self._log_prob_in_bits
        )
    @property
    def num_bits(self):
        """返回模式中每个随机二进制向量的维度。"""
        return self._num_bits

    @property
    def target_size(self):
        """返回目标张量的维度。"""
        return self._num_bits + 1

    @property
    def time_average_cost(self):
        """返回是否在损失计算时平均时间维度。"""
        return self._time_average_cost

    @property
    def log_prob_in_bits(self):
        """返回是否以比特为单位表达对数概率。"""
        return self._log_prob_in_bits

    @property
    def batch_size(self):
        """返回当前批次的大小。"""
        return self._batch_size

    def to_human_readable(self, data, model_output=None, whole_batch=False):
        obs = data.observations
        # 使用 TensorFlow 的 round 函数来替代 NumPy 的 round
        unnormalised_num_reps_flag = tf.round(self._unnormalise(obs[:, :, -1:]))
        obs = np.concatenate([obs[:, :, :-1], unnormalised_num_reps_flag.numpy()], axis=2)
        data = data._replace(observations=obs)
        return bitstring_readable(data, self._batch_size, model_output, whole_batch)