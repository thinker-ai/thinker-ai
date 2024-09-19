# train_model.py
import tensorflow as tf
import numpy as np
import datetime
from simple_model_with_cosine import SimpleModel

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 预处理数据
train_images = train_images.reshape(-1, 28*28).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28*28).astype('float32') / 255.0

# 将标签转换为 one-hot 编码
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# 创建训练和验证数据集
batch_size = 64

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size)

validation_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = validation_ds.batch(batch_size)