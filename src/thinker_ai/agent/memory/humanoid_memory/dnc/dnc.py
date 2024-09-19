import collections
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc import access

# 定义 DNCState，使用 namedtuple 便于管理 DNC 的状态
DNCState = collections.namedtuple('DNCState', (
    'access_output', 'access_state', 'controller_state'))


class DNC(tf.keras.layers.Layer):
    """区分性神经计算机（DNC）的核心模块，适用于 TensorFlow 2.x。

    该模块包含一个控制器（LSTM）和一个内存访问模块。它接收输入，经过控制器和内存模块处理，
    生成最终输出。
    """

    def __init__(self, access_config, controller_config, output_size, clip_value=None, name='dnc'):
        """初始化 DNC 核心模块。

        Args:
          access_config: 访问模块配置的字典。
          controller_config: 控制器（LSTM）配置的字典。
          output_size: DNC 核心模块的输出维度大小。
          clip_value: 若指定，则将控制器和核心输出的值进行裁剪。
          name: 模块名称（默认为 'dnc'）。
        """
        super(DNC, self).__init__(name=name)

        # 内存访问模块
        self._access = access.MemoryAccess(**access_config)

        # 控制器：使用 LSTM 作为控制器
        self._controller = tf.keras.layers.LSTMCell(controller_config['units'])

        # 输出维度大小
        self._output_size = output_size

        # 裁剪输出值的最大绝对值，防止梯度爆炸（如果指定）
        self._clip_value = clip_value

        # 初始化状态（将在每次调用前设置）
        self.prev_state = None

    def call(self, inputs, training=False):
        """执行 DNC 核心运算。

        Args:
          inputs: 输入张量。
          training: 标志指示当前是否为训练模式。

        Returns:
          返回一个元组 (output, next_state)，其中：
            - output 是 DNC 核心的输出。
            - next_state 是处理输入后的更新状态。
        """
        if self.prev_state is None:
            raise ValueError("prev_state 未初始化，请调用 `get_initial_state` 方法。")

        # 第一步：运行控制器（LSTM）
        controller_output, controller_state = self._controller(inputs, self.prev_state.controller_state)

        # 第二步：使用控制器的输出与内存进行交互（访问模块）
        access_output, access_state = self._access(controller_output, self.prev_state.access_state)

        # 第三步：将控制器的输出与内存的输出结合
        final_output = tf.concat([controller_output, access_output], axis=1)

        # 若裁剪值被设置，执行输出裁剪，防止数值爆炸
        if self._clip_value is not None:
            final_output = tf.clip_by_value(final_output, -self._clip_value, self._clip_value)

        # 更新 prev_state 以备下次调用
        self.prev_state = DNCState(access_output, access_state, controller_state)

        # 返回输出和更新后的状态
        return final_output, self.prev_state

    @property
    def state_size(self):
        """返回 DNC 状态的形状（包括控制器和访问模块的状态）。

        Returns:
          DNCState: 包含访问输出、访问状态和控制器状态形状的命名元组。
        """
        return DNCState(
            access_output=tf.TensorShape([self._access.output_size]),
            access_state=self._access.state_size,
            controller_state=self._controller.state_size)

    @property
    def output_size(self):
        """返回输出的形状。

        Returns:
          TensorShape: 经过控制器和访问模块组合后的输出形状。
        """
        return self._output_size

    def get_initial_state(self, batch_size):
        """返回 DNC 的初始状态。

        Args:
          batch_size: 初始状态的批次大小。

        Returns:
          The initial state of the DNC as a DNCState tuple.
        """
        # 初始化控制器状态（LSTM 状态）
        controller_state = self._controller.get_initial_state(batch_size=batch_size, dtype=tf.float32)

        # 初始化访问状态（内存状态）
        access_state = self._access.get_initial_state(batch_size)

        # 访问输出初始化为零
        access_output = tf.zeros([batch_size, self._access.output_size], dtype=tf.float32)

        # 初始化 prev_state 为初始状态
        self.prev_state = DNCState(access_output, access_state, controller_state)

        return self.prev_state

    def build(self, input_shape):
        """构建 DNC 的计算图（适用于 tf.keras）。

        Args:
          input_shape: 传递给 DNC 的输入形状。
        """
        # 构建控制器
        self._controller.build(input_shape)

        # 构建访问模块
        self._access.build([input_shape[0], self._controller.units])
