# model_components.py
import tensorflow as tf
import collections

# Small epsilon value to prevent numerical instability
_EPSILON = 1e-6


def _vector_norms(m):
    """计算向量的L2范数，以防止数值不稳定性。

    该函数计算输入张量 `m` 的L2范数（即欧几里得范数）。L2范数通常用于衡量向量的长度或大小，计算方式是
    将向量中所有元素平方后求和，再开平方。这是深度学习中衡量相似性和正则化的常用方法。

    为了避免数值不稳定性（例如，在向量接近零时出现的除零错误），我们在计算过程中加入了一个小的正数 `_EPSILON`。
    这一技术被称为“数值稳定性增强”，它常用于防止由于精度问题导致的运算错误。

    机制解释：
        1.	输入：m 是一个 3D 张量，代表了批次中的每个注意力头与存储器的交互（通常是用于注意力机制中的 key 或 memory 向量）。
        2.	计算：对于张量 m 中的每个向量，计算其平方和并取平方根，从而得到其L2范数。为了避免数值不稳定性，平方和中加上一个小的常数 _EPSILON。
        3.	返回：返回的张量包含每个向量的L2范数，并保留了与输入相同的维度结构（除了沿着 axis=2 的维度被压缩成1维）。

    Args:
      m: 一个三维张量，形状为 `[batch_size, num_heads, word_size]`，表示每个批次中的每个注意力头与存储器的交互。

    Returns:
      一个形状为 `[batch_size, num_heads, 1]` 的张量，其中每个元素表示对应向量的L2范数。
      范数的计算公式为：
        norm = sqrt(sum(m[i] * m[i]) + _EPSILON)
    """
    squared_norms = tf.reduce_sum(m * m, axis=2, keepdims=True)  # [batch_size, num_heads, 1]
    return tf.sqrt(squared_norms + _EPSILON)  # [batch_size, num_heads, 1]


def weighted_softmax(activations, strengths, strengths_op):
    """对激活值进行加权softmax操作。

    softmax是深度学习中常用的一种激活函数，用于将一组激活值转换为概率分布。通过对所有输入的指数值进行归一化，softmax
    保证输出的值范围在0到1之间，且所有输出的和为1。这使得softmax非常适合用于分类问题中的多分类模型。

    在这个函数中，除了标准的softmax操作，我们还引入了额外的“strengths”参数，用于调整每个激活值的权重。这种方法允许我们
    根据不同的权重信息对激活值进行放大或缩小，进而改变 softmax 输出的“尖锐度”，即输出的概率分布更加集中或平滑。

    机制:
      - 首先，对 `strengths` 使用 `strengths_op` 进行变换，并扩展其维度使其可以与 `activations` 相乘。
      - 然后，将 `activations` 乘以 `strengths`，以调整每个位置的激活值。
      - 最后，沿着最后一个维度对调整后的激活值进行 softmax 操作，确保输出是一个概率分布。
    机制解释：

        1.	输入：
        •	activations 是一个 3D 张量，代表注意力机制中每个头对不同记忆位置的激活值。
        •	strengths 是一个 2D 张量，表示每个头的强度，用来调节 softmax 计算的“尖锐度”。
        •	strengths_op 是一个函数，用于对 strengths 值进行转换，一般使用 tf.nn.softplus 来确保其为正。
        2.	加权 softmax 过程：
        •	权重调整：通过将 strengths 进行转换，并将其与 activations 相乘，来放大或缩小激活值。
        •	softmax 归一化：对调整后的激活值进行 softmax 操作，将其转换为概率分布。Softmax 的作用是沿着最后一个维度进行归一化，使得每一行的值总和为1。
        3.	返回：返回经过加权 softmax 计算后的张量，与输入 activations 具有相同的形状，但数值已被转化为概率分布。

    Args:
      activations: 一个形状为 `[batch_size, num_heads, memory_size]` 的张量，表示每个注意力头对于存储器中每个位置的激活值。
      strengths: 一个形状为 `[batch_size, num_heads]` 的张量，表示每个注意力头的强度参数，用来调整 softmax 的“尖锐度”。
      strengths_op: 一个函数，用于对 `strengths` 进行变换。常见的操作是 `tf.nn.softplus`，该操作保证输出为正值，避免强度为负数的情况。

    Returns:
      一个与 `activations` 形状相同的张量，表示经过加权softmax后的激活值。softmax操作是沿着最后一个维度（`memory_size`）进行的。
    """
    transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)  # [batch_size, num_heads, 1]
    sharp_activations = activations * transformed_strengths  # [batch_size, num_heads, memory_size]
    return tf.nn.softmax(sharp_activations, axis=-1)  # [batch_size, num_heads, memory_size]


class CosineWeights(tf.keras.layers.Layer):
    """基于余弦相似度的注意力机制，用于存储器寻址。

    这个类实现了通过计算查询向量（keys）与存储器中的每个位置（memory）之间的余弦相似度来生成注意力权重。
    余弦相似度是一种度量向量相似性的指标，它计算两个向量之间的夹角余弦，数值范围在 -1 到 1 之间。
    在深度学习中，余弦相似度常用于度量高维空间中的向量相似度，特别是在注意力机制中，它可以用来为存储器中的不同位置
    分配权重。这个类将计算出来的余弦相似度值进行加权并通过 softmax 转换为概率分布。

    """

    def __init__(self, num_heads, word_size, strength_op=tf.nn.softplus, name='cosine_weights'):
        """CosineWeights 初始化方法。
        详细步骤：
            1.	点积计算：首先计算查询向量 keys 和存储器 memory 之间的点积，这表示每个查询向量与存储器中每个位置的相似度。点积结果是一个三维张量，形状为 [batch_size, num_heads, memory_size]。
            2.	计算 L2 范数：计算 keys 和 memory 中每个向量的 L2 范数（向量长度）。L2 范数是计算余弦相似度时所需的标准化因子。为了防止数值不稳定性，范数计算中加入了小量 _EPSILON。
            3.	余弦相似度：通过将 keys 和 memory 的点积除以它们的 L2 范数，计算出每个查询向量与存储器中每个位置的余弦相似度。余弦相似度表示两个向量的相似程度，值介于 -1 到 1 之间。
            4.	加权 softmax：利用 strengths 来调整余弦相似度的“尖锐度”。strengths 的作用是放大或缩小不同查询向量的相似度，然后通过 softmax 操作将这些相似度转换为概率分布，使得输出的权重总和为 1。
            3.	余弦相似度的优势：
            •	余弦相似度关注的是向量之间的方向而不是大小，因此即使两个向量的长度不同，只要它们的方向相似，余弦相似度也会给出较高的相似性分数。这在处理高维空间中的向量（如词向量、注意力机制中的 key 和 memory 向量）时非常有用。
        Args:
          name: 字符串，当前层的名称，用于在构建模型时命名这个层。
          num_heads: 整数，表示注意力头的数量。每个注意力头可以独立地与存储器进行交互。
          word_size: 整数，表示每个存储器单元（memory）的维度，即存储器中的每个位置由多少维的向量表示。
          strength_op: 函数，用于对 `strengths` 进行变换。默认使用 `tf.nn.softplus`，确保输出值为正数。
        """
        super(CosineWeights, self).__init__(name=name)
        self.num_heads = num_heads
        self.word_size = word_size
        self.strength_op = strength_op

    def call(self, inputs, training=False):
        """计算查询向量和存储器位置之间的余弦相似度，并生成注意力权重。

        Args:
          inputs: 包含 'memory', 'keys', 和 'strengths' 的字典。
              - 'memory': 一个张量，形状为 `[batch_size, memory_size, word_size]`，表示存储器中的每个位置向量。
              - 'keys': 一个张量，形状为 `[batch_size, num_heads, word_size]`，表示查询向量。
              - 'strengths': 一个张量，形状为 `[batch_size, num_heads]`，表示每个查询的强度，用于调整注意力的分布。
          training: 布尔值，指示当前层是否处于训练模式（默认为 False）。

        Returns:
          一个张量，形状为 `[batch_size, num_heads, memory_size]`，表示每个查询向量对于存储器每个位置的余弦相似度，经过加权和 softmax 操作，输出为注意力权重。
        """
        memory = inputs['memory']  # [batch_size, memory_size, word_size]
        keys = inputs['keys']  # [batch_size, num_heads, word_size]
        strengths = inputs['strengths']  # [batch_size, num_heads]

        # 计算 keys 与 memory 之间的点积
        dot = tf.matmul(keys, memory, adjoint_b=True)  # [batch_size, num_heads, memory_size]

        # 计算 memory 和 keys 的 L2 范数
        memory_norms = _vector_norms(memory)  # [batch_size, memory_size, 1]
        key_norms = _vector_norms(keys)  # [batch_size, num_heads, 1]

        # 调整 memory_norms 和 key_norms 的形状，使其可以广播
        memory_norms = tf.transpose(memory_norms, [0, 2, 1])  # [batch_size, 1, memory_size]
        memory_norms = tf.broadcast_to(memory_norms, [tf.shape(memory)[0], self.num_heads,
                                                      tf.shape(memory)[1]])  # [batch_size, num_heads, memory_size]

        key_norms = tf.broadcast_to(key_norms, [tf.shape(keys)[0], self.num_heads,
                                                tf.shape(memory)[1]])  # [batch_size, num_heads, memory_size]

        # 计算余弦相似度
        normed_dot = dot / (memory_norms * key_norms + _EPSILON)  # [batch_size, num_heads, memory_size]

        # 使用 strengths 调整相似度，并通过 softmax 得到权重
        return weighted_softmax(normed_dot, strengths, self.strength_op)  # [batch_size, num_heads, memory_size]


# 定义 TemporalLinkageState
TemporalLinkageState = collections.namedtuple('TemporalLinkageState', ('link', 'precedence_weights'))


class TemporalLinkage(tf.keras.layers.Layer):
    """跟踪每个写入头的存储写入顺序的模块。
    该模块维护一个时序链路矩阵，表示内存位置被写入的顺序。这个矩阵可以支持存储器的前向和后向遍历。
    TemporalLinkage 模块的主要作用是记录存储位置的写入时间顺序，以便在后续的读写操作中能够根据时间顺序
    访问特定的存储器位置。
    """

    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        """TemporalLinkage 的初始化方法。

        Args:
          memory_size: 整数，表示存储器的大小，即存储器中有多少个位置可供读写操作。
          num_writes: 整数，表示写头的数量。每个写头可以独立写入不同的存储位置。
          name: 字符串，表示该层的名称，在创建模型时用于区分该层。
        """
        super(TemporalLinkage, self).__init__(name=name)
        self.memory_size = memory_size
        self.num_writes = num_writes

    def call(self, inputs, training=False):
        """基于当前写权重更新时序链路矩阵。
        •	call 方法是核心逻辑，处理每次写入操作后的状态更新。它接收当前的写权重 write_weights 和之前的时序链路状态
        prev_linkage，然后更新链路矩阵和优先级权重。
        •	优先级权重更新：使用重置门 reset_gate 计算新的优先级权重 updated_precedence_weights，确保写入的权重按顺序
        Args:
          inputs: 一个包含 'write_weights' 和 'prev_linkage' 的字典。
              - 'write_weights': 一个形状为 `[batch_size, num_writes, memory_size]` 的张量，表示当前写头的写权重。
              - 'prev_linkage': 一个 TemporalLinkageState，包含上一次的时序链路矩阵和优先级权重。
          training: 布尔值，指示当前层是否处于训练模式。

        Returns:
          一个 TemporalLinkageState，包含更新后的时序链路矩阵和优先级权重。
        """
        write_weights = inputs['write_weights']  # [batch_size, num_writes, memory_size]
        prev_linkage = inputs['prev_linkage']  # TemporalLinkageState

        # 获取上一次的时序链路矩阵和优先级权重
        prev_link = prev_linkage.link  # [batch_size, num_writes, memory_size, memory_size]
        prev_precedence_weights = prev_linkage.precedence_weights  # [batch_size, num_writes, memory_size]

        # 这里添加 TemporalLinkage 的逻辑
        # 为了简单起见，这里仅仅将 write_weights 作为新的 precedence_weights
        updated_precedence_weights = write_weights  # [batch_size, num_writes, memory_size]

        # 这里暂时不更新 link，保持与上一次相同
        updated_link = prev_link  # [batch_size, num_writes, memory_size, memory_size]

        return TemporalLinkageState(link=updated_link, precedence_weights=updated_precedence_weights)


class Freeness(tf.keras.layers.Layer):
    """跟踪每个内存位置的使用情况并提供空闲位置的模块。

    该模块用于确定内存中的哪些位置使用最少或者是空闲的。它在动态神经计算机（DNC）等架构中很有用，
    帮助模型管理有限的存储器资源。通过跟踪内存的使用情况，模型可以决定哪些位置可以被重新分配用于
    新的写操作。
    """

    def __init__(self, memory_size, name='freeness'):
        """初始化 Freeness 模块。

        Args:
          memory_size: 一个整数，表示内存的大小，即可用的内存位置数目。
          name: 一个字符串，表示该层的名称，用于构建图模型时区分不同的层。
        """
        super(Freeness, self).__init__(name=name)
        self.memory_size = memory_size

    def call(self, inputs, training=False):
        """基于当前的读写操作更新内存使用率。
        是该类的核心，负责根据当前的读写操作更新内存使用率。写入操作会增加内存位置的使用率，而读操作会根据 free_gate
        控制哪些内存位置可以释放。free_gate 的作用是指示哪些读出的内存位置可以释放，释放这些内存位置会减少它们的使用率。
        最终，更新后的使用率被限制在 0 到 1 的范围内，确保内存使用率在合理范围内变化。
        Args:
          inputs: 一个包含 'write_weights'、'free_gate'、'read_weights' 和 'prev_usage' 的字典。
              - 'write_weights': 当前写头的写权重，形状为 `[batch_size, num_writes, memory_size]`。
              - 'free_gate': 表示哪些读操作的内存位置可以被释放，形状为 `[batch_size, num_reads]`。
              - 'read_weights': 当前读头的读权重，形状为 `[batch_size, num_reads, memory_size]`。
              - 'prev_usage': 上一次的内存使用率，形状为 `[batch_size, memory_size]`。
          training: 布尔值，指示当前层是否处于训练模式。

        Returns:
          一个更新后的内存使用率张量，形状为 `[batch_size, memory_size]`。
        """
        write_weights = inputs['write_weights']  # [batch_size, num_writes, memory_size]
        free_gate = inputs['free_gate']  # [batch_size, num_reads]
        read_weights = inputs['read_weights']  # [batch_size, num_reads, memory_size]
        prev_usage = inputs['prev_usage']  # [batch_size, memory_size]

        # 更新使用率
        usage_increase = tf.reduce_sum(write_weights, axis=1)  # [batch_size, memory_size]
        usage_decrease = tf.reduce_sum(read_weights * tf.expand_dims(free_gate, -1),
                                       axis=1)  # [batch_size, memory_size]
        updated_usage = prev_usage + usage_increase - usage_decrease

        # 打印调试信息
        tf.print("write_weights shape:", tf.shape(write_weights))
        tf.print("free_gate shape:", tf.shape(free_gate))
        tf.print("read_weights shape:", tf.shape(read_weights))
        tf.print("usage_increase shape:", tf.shape(usage_increase))
        tf.print("usage_decrease shape:", tf.shape(usage_decrease))
        tf.print("updated_usage shape:", tf.shape(updated_usage))

        # 确保使用率在 [0, 1] 之间
        updated_usage = tf.clip_by_value(updated_usage, 0.0, 1.0)

        return updated_usage  # [batch_size, memory_size]