import tensorflow as tf
from model import create_model_cnn
import numpy as np
from read_data import get_data
import os
import matplotlib.pylab as plt


LOG_DIR = './model_dir'
TIME_STEP = 5  
BATCH_SIZE = 1000 
CHANNEL = 5  #通道数
LEARNING_RATE = 0.001  # 学习率
EPOCH = 1000  # 训练的轮数
file_name = 'USDT_BTC_2018.csv'  # 测试文件名
test_x, test_y, mean_data, std_data = get_data(file_name, TIME_STEP)

test_y = np.array(test_y)
test_x = np.array(test_x)

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
checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
saver.restore(sess, checkpoint_path)
dst = None

print("开始测试")
pred = sess.run([net], feed_dict={input_x: test_x})[0]
pred = pred[:, 0]
plot_len =  100
index = [i for i in range(plot_len)]
pred = pred[-plot_len:] * std_data[3] + mean_data[3]
test_y = test_y[-plot_len:] * std_data[3] + mean_data[3]
plt.style.use("ggplot")
fig = plt.figure(figsize=(16, 6))

plt.plot(index, pred)
plt.plot(index, test_y)
plt.xlabel('time')
plt.ylabel('price')
plt.savefig('fig1.svg', dpi=300)
plt.show()