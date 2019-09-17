import tensorflow as tf
from tensorflow.contrib import slim


def create_model_cnn(input_x):
    net_cnn_1 = tf.layers.conv1d(input_x, filters=16, kernel_size=5, strides=1, padding='same')
    net_cnn_2 = tf.layers.conv1d(net_cnn_1, filters=32, kernel_size=4, strides=1, padding='same')
    net_cnn_3 = tf.layers.conv1d(net_cnn_2, filters=64, kernel_size=3, strides=1, padding='same')
    # net = tf.concat([net_cnn_1, net_cnn_2, net_cnn_3], axis=1)
    net = tf.layers.flatten(net_cnn_3)
    # net = slim.dropout(net, keep_prob)
    net = slim.fully_connected(net, 1, activation_fn=None)
    return net

