import tensorflow as tf
from tensorflow.contrib import slim


def create_model_cnn(input_x, keep_prob):
    net = tf.layers.conv1d(input_x, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    net = tf.layers.conv1d(net, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    net = tf.layers.conv1d(net, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    net = tf.layers.flatten(net)
    net = slim.dropout(net, keep_prob)
    net = tf.layers.dense(net, 100)
    net = tf.layers.dense(net, 3)
    return net

