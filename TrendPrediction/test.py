import tensorflow as tf
from model import create_model_cnn
import numpy as np
from read_data import get_data
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
LOG_DIR = './model_dir'
TIME_STEP = 5  # 滑动窗口size
BATCH_SIZE = 1000  # 每次送入训练的样本数量
CHANNEL = 5  #送入训练的通道数
LEARNING_RATE = 0.001  # 学习率
EPOCH = 1000  # 训练的轮数
file_name = 'USDT_BTC_2018.csv'  # 测试文件名
test_x, test_y = get_data(file_name, TIME_STEP)
test_y = np.array(test_y).astype(np.int)
test_x = np.array(test_x)

graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, [None, TIME_STEP, CHANNEL])
    input_y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32)
    # 模型的返回为相关的预测值
    net = create_model_cnn(input_x,keep_prob=keep_prob)
    # 损失函数
    predicted_labels = tf.argmax(net, axis=1)

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=input_y))
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, tf.cast(input_y, dtype=tf.int64)),
                                 dtype=tf.float32),
                         name='acc')
    # cost = tf.reduce_mean(tf.square(tf.reshape(net, [-1]) - tf.reshape(input_y, [-1])))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

sess = tf.Session(graph=graph)
_ = sess.run([init])
checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
saver.restore(sess, checkpoint_path)
dst = None

print("开始测试")
pred = sess.run([predicted_labels], feed_dict={input_x: test_x,
                                      input_y: test_y,
                                      keep_prob: 1.0})[0]

score = accuracy_score(test_y, pred)
print("测试准确率：{:.3f}".format(score))
print("召回率:{}".format(recall_score(test_y, pred, average="macro")
))
print("卡帕系数:{}".format(cohen_kappa_score(test_y, pred)))
print("f1:{}".format(f1_score(test_y, pred, average="macro")))