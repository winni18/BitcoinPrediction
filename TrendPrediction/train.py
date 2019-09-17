"""
训练模型文件
"""
import tensorflow as tf
from model import create_model_cnn
import numpy as np
from read_data import get_data, balance_data
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random
LOG_DIR = './bitcorn_model2'
TIME_STEP = 6  # 滑动窗口大小
BATCH_SIZE = 4096  # 每次送入训练的样本数量

CHANNEL = 5  # 送入训练练的维数
LEARNING_RATE = 0.003  # 学习率
EPOCH = 20  # 训练的轮数
file_name = 'USDT_BTC 5min(2015-2018).csv'  # 训练文件名
data_x, data_y = get_data(file_name, TIME_STEP)
data_y = np.array(data_y).astype(np.int)
data_x = np.array(data_x)
# 割训练集与测试集
len_train = int(len(data_y) * 0.7)
len_train_test = int(len(data_y)*0.85)
train_x = data_x[:len_train]
train_y = data_y[:len_train]
test_x = data_x[len_train:]
test_y = data_y[len_train:]
val_x = data_x[len_train_test:]
val_y = data_y[len_train_test:]
print(train_x.shape)
graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, [None, TIME_STEP, CHANNEL])
    input_y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32)
    # 模型的返回为相关的预测值
    net = create_model_cnn(input_x, keep_prob=keep_prob)
    for v in tf.trainable_variables():
        tf.summary.histogram(v.name, v)
    predicted_labels = tf.argmax(net, axis=1)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=input_y))
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, tf.cast(input_y, dtype=tf.int64)),
                                 dtype=tf.float32),
                         name='acc')
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', acc)
    summary_op = tf.summary.merge_all()
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=config)
train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
_ = sess.run([init])
# index = [i for i in range(len(train_y))]
index = balance_data(train_y)
print(len(index))
step = 0
print("开始训练")
for i in range(EPOCH):
    random.shuffle(index)
    for j in range(0, len(index), BATCH_SIZE):
        step += 1
        start = j
        end = min(j+BATCH_SIZE, len(index))
        batch_x = train_x[index[start:end]]
        batch_y = train_y[index[start:end]]
        _, loss, train_acc, summary_str = sess.run([train_op, cost, acc, summary_op], feed_dict={input_x: batch_x,
                                                                                                 input_y: batch_y,
                                                                                                 keep_prob: 0.5})
        train_writer.add_summary(summary_str, step)
        val_acc = sess.run([acc], feed_dict={input_x: val_x,
                                             input_y: val_y,
                                             keep_prob: 1.0})[0]
        print("epoch:{}, step:{}, train loss:{:.3f}, train_acc:{:.3f}, val acc:{:.3f}".format(i, step, loss,
                                                                                              train_acc, val_acc))
print("开始测试")
pred = sess.run([predicted_labels], feed_dict={input_x: test_x,
                                               input_y: test_y,
                                               keep_prob: 1.0})[0]
score = accuracy_score(test_y, pred)
print("后面30%数据测试，准确率:{:.3f}".format(score))
print("召回率:{}".format(recall_score(test_y, pred, average="macro")))
print("卡帕系数:{}".format(cohen_kappa_score(test_y, pred)))
print("f1:{}".format(f1_score(test_y, pred, average="macro")))
# 保存模型
checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
saver.save(sess, checkpoint_path)