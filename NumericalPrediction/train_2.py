
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
TIME_STEP = 5  # 滑动窗口大小
BATCH_SIZE = 5000  # 每次送入训练的样本数量

CHANNEL = 5  # 送入训练练的维数
LEARNING_RATE = 0.001  # 学习率
EPOCH = 100  # 训练的轮数
file_name = 'USDT_BTC_2017.csv'  
data_x, data_y, mean_data, std_dat = get_data(file_name, TIME_STEP)

data_y = np.array(data_y)
print(data_y.shape)
data_x = np.array(data_x)

# 分割训练集与测试集
len_train = int(len(data_y) * 0.7)
len_train_test = int(len(data_y)*0.85)
train_x = data_x[:len_train]
train_y = data_y[:len_train]
test_x = data_x[len_train:len_train_test]
test_y = data_y[len_train:len_train_test]
val_x = data_x[len_train_test:]
val_y = data_y[len_train_test:]

print(test_y.shape)
print(test_x.shape)
graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(tf.float32, [None, TIME_STEP, CHANNEL])
    input_y = tf.placeholder(tf.float32, [None])
    # 模型的返回为相关的预测值
    net = create_model_cnn(input_x)

    cost = tf.reduce_mean(tf.square(tf.reshape(net, [-1]) - tf.reshape(input_y, [-1])))
    tf.summary.scalar('loss_function', cost)
    merged_summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss_function')])
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()

sess = tf.Session(graph=graph)
_ = sess.run([init])

summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
step = 0
print("开始训练")
for i in range(EPOCH):
    for j in range(0, len(train_y), BATCH_SIZE):
        step += 1
        start = j
        end = min(j+BATCH_SIZE, len(train_y))
        batch_x = train_x[start:end]
        batch_y = train_y[start:end]

        _, loss, summary_str = sess.run([train_op, cost, merged_summary_op], feed_dict={input_x: batch_x,
                                                                                        input_y: batch_y})
        summary_writer.add_summary(summary_str, step)

        print("epoch:{}, step:{}, train loss:{:.3f}".format(i, step, loss))

checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
saver.save(sess, checkpoint_path)