import os
import cv2
import random
import pickle
import numpy as np
from DQN import Game
# TensorFlow
# Google的深度学习框架
# TensorBoard可视化很方便
# 数据和模型并行化好，速度快
import tensorflow as tf
from collections import deque
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.compat.v1.disable_v2_behavior()
# 游戏名
GAME = 'Jump'
# 游戏操作个数(无动作、向上跳、向左走、向右走)
ACTIONS = 3
# 折扣系数
GAMMA = 0.8
# 尝试游戏帧数,迭代前观察帧数
OBSERVE = 2520
# 迭代次数，训练
EXPLORE = 500000
# 贪婪系数
FINAL_EPSILON = 0.01  # 最终值
INITIAL_EPSILON = 0.01  # 初始值
# 训练记忆容量
REPLAY_MEMORY = 50000
# 单次迭代状态数
BATCH = 64
# 单步走帧数
FRAME_PER_ACTION = 1


# w服从正态分布取随机数
def weights_variable(shape):
    # 截断的产生正态分布的函数
    # shape为生成张量的维度，即多维数组，参数为维度值
    # stddev为标准差
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
    # 用于生成一个初始值为initial的变量。必须指定初始化值。
    return tf.Variable(initial)


# b初始化
def bias_variable(shape):
    # TensorFlow中创建常量的函数
    # value为常量，shape为创建多维数组，其中各个数据均为value常量值
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# 卷积计算的函数
def conv2d(x, w, stride):
    # x为输入的要做卷积的图片，要求为一个张量
    # w为卷积核，要求也是一个张量
    # stride为卷积时在图像每一维的步长
    # padding为填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


# 池化方法
# 压缩图像特征，方便优化减少运算
def max_pool_2x2(x):
    # x为输入图片特征图数据
    # ksize为池化窗口大小[单次抓取数据，高，宽，渠道数]
    # stride为窗口在维度上滑动的步长
    # padding为填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构造卷积神经网络
def create_network():
    w_conv1 = weights_variable([8, 8, 1, 32])
    b_conv1 = bias_variable([32])

    w_conv2 = weights_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    # w_conv3 = weights_variable([3, 3, 64, 64])
    # b_conv3 = bias_variable([64])

    W_fc1 = weights_variable([576, 512])
    b_fc1 = bias_variable([512])
    # w_fc1 = weights_variable([1600, 512])
    # b_fc1 = weights_variable([512])

    w_fc2 = weights_variable([512, ACTIONS])
    b_fc2 = weights_variable([ACTIONS])

    # 类型规范，存储数据
    tf.compat.v1.disable_eager_execution()
    # s = tf.compat.v1.placeholder('float', [None, 80, 80, 4])
    s = tf.compat.v1.placeholder('float', [None, 80, 80, 1])

    # 第一次卷积计算
    # tf.nn.relu为线性整流函数，作为Relu激活层
    # tf.nn.relu()函数的目的是，将输入小于0的值幅值为0，输入大于0的值不变。
    h_conv1 = tf.nn.relu(conv2d(s, w_conv1, 4) + b_conv1)
    # 池化
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # tf.reshape函数用于对输入tensor进行维度调整，但是这种调整方式并不会修改内部元素的数量以及元素之间的顺序
    h_pool3_flat = tf.reshape(h_pool2, [-1, 576])
    # h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    # tf.matmul为矩阵相乘函数
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    # h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, w_fc2) + b_fc2
    return s, readout, h_fc1


# 核心训练模块
def train_network(s, readout, h_fc1, sess):

    score = 0
    last_reward = 0
    frequency = 0
    # 所有可能情况的行动
    behavior = [0, 0,
                2, 0,
                1, 1, 0,
                1, 2, 0,
                1, 0, 1, 0,
                1, 0, 0, 0,
                1, 0, 2, 1, 0,
                1, 0, 2, 2, 0,
                1, 0, 2, 0, 0, 0,
                1, 0, 2, 0, 2, 0,
                1, 0, 2, 0, 1, 0, 0,
                1, 0, 2, 0, 1, 2, 0,
                1, 0, 2, 0, 1, 1, 0, 0, 0]
    a = tf.compat.v1.placeholder('float', [None, ACTIONS])
    y = tf.compat.v1.placeholder('float', [None])
    # tf.multiply逐个相乘
    # tf.reduce_mean计算单轴上的平均值，可用于降维
    # tf.square计算平方函数
    # tf.train.AdamOptimizer.minimize是TensorFlow里面的一个优化器，对损失函数进行优化，它可以用来训练神经网络模型。
    readout_action = tf.reduce_mean(tf.multiply(readout, a), axis=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    # train_step = tf.compat.v1.train.GradientDescentOptimizer(0.0001).minimize(cost)
    train_step = tf.compat.v1.train.AdamOptimizer(1e-8).minimize(cost)

    Game.game_start()
    D = deque()

    # 日志
    a_file = open("log_"+GAME+"/readout.txt", 'w')
    h_file = open("log_"+GAME+"/hidden.txt", 'w')

    # 初始化行动并尝试走一步
    do_nothing = np.zeros(ACTIONS)
    # 转化为灰度图,二值化
    x_t, r_0, terminal, score = Game.frame_step(do_nothing, score)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    x_t = np.reshape(x_t, (80, 80, 1))
    s_t = x_t
    last_reward = r_0
    # img = Image.fromarray(s_t).convert('RGB')  # 将数组转化回图片
    # img.save("out.bmp")  # 将数组保存为图片

    # 保存模型初始化
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    # 加载训练后的神经网络
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("找不到已有神经网络的权重值")

    epsilon = INITIAL_EPSILON
    t = 0

    # 导入中断保存的行动
    # D = pickle.load(open('saved_deque', 'rb'))
    if len(D) != 0:
        for i in range(0, 3):
            D.pop()
        if len(D) != 0:
            last_reward = D[-1][2]
            s_t = D[-1][0]
            t += len(D)

    # # 存放实际的状态-动作-Q值
    # s_j_batch = []
    # a_batch = []
    # r_batch = []
    # y_batch = []
    # s_j1_batch = []

    while True:
        # feed_dict为占位符赋值，当前为状态赋值
        # 预测结果(当前状态不同行为action的回报)
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            # if t < OBSERVE:
            #     action_index = behavior[t % len(behavior)]
            #     a_t[action_index] = 1
            # else:
            # 尝试冒险
            # if random.random() <= epsilon:
            #     print("----------Random Action----------")
            #     action_index = random.randrange(ACTIONS)
            #     a_t[action_index] = 1
            # else:
                # np.argmax返回最大值的索引值
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # 贪婪系数衰减
        # if epsilon > FINAL_EPSILON and t > OBSERVE:
        #     epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EXPLORE

        x_t1_colored, r_t, terminal, score = Game.frame_step(a_t, score)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        # s_t1 = np.stack((x_t1, x_t1, x_t1, x_t1), axis=2)
        s_t1 = x_t1

        # 如果进程出错，退出训练
        if last_reward == 1.0 and r_t == 1.0:
            None
        elif last_reward == r_t:
            frequency += 1
            if frequency >= 4:
                with open("saved_deque", "wb") as file:
                    file.truncate(0)
                    pickle.dump(D, file)
                    pass
                break
        else:
            last_reward = r_t
            frequency = 0

        D.append((s_t, a_t, r_t, s_t1, terminal))
        # 将数组转化回图片
        # img = Image.fromarray(s_t1).convert('RGB')
        # 将数组保存为图片
        # img.save("out.bmp")

        # 如果训练超过设置容量，抛弃最早的记录
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            # 获取batch = 32个保存的参数集
            minibatch = random.sample(D, BATCH)

            # 获取j时刻batch(32)个状态state
            s_j_batch = [d[0] for d in minibatch]
            # 获取batch(32)个行动action
            a_batch = [d[1] for d in minibatch]
            # 获取保存的batch(32)个奖励reward
            r_batch = [d[2] for d in minibatch]
            # 获取保存的j + 1时刻的batch(32)个状态state
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []

            readout_j1_batch = sess.run(readout, feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            if t % 5000 == 0:
                print(r_batch)
                print(y_batch)

            # 训练神经网络模型
            train_step.run(feed_dict={
                            y: y_batch,
                            a: a_batch,
                            s: s_j_batch
            })

        # 实际加快训练的代码
        # if t >= 63:
        #     # 随机从记忆池中取BATCH个元素训练神经网络
        #     minibatch = random.sample(D, BATCH)
        #     # 拓展实际Q值表
        #     for el in minibatch:
        #         flag = True
        #         for i in range(0, len(s_j_batch)):
        #             if np.array_equal(s_j_batch[i], el[0]) and np.array_equal(a_batch[i], el[1]):
        #                 flag = False
        #         if flag:
        #             s_j_batch.append(el[0])
        #             a_batch.append(el[1])
        #             r_batch.append(el[2])
        #             y_batch.append(el[2])
        #             s_j1_batch.append(el[3])
        #     # 更新实际Q值表
        #     while True:
        #         change_math = 0
        #         for i in range(0, len(s_j_batch)):
        #             if y_batch[i] != -1.0:
        #                 y1_batch = []
        #                 for m in range(0, len(s_j_batch)):
        #                     if np.array_equal(s_j_batch[m], s_j1_batch[i]):
        #                         y1_batch.append(y_batch[m])
        #                 # 当下一状态所有动作录入表内进行更新
        #                 if len(y1_batch) == ACTIONS:
        #                     change = r_batch[i] + GAMMA * np.max(y1_batch)
        #                     if y_batch[i] != change:
        #                         y_batch[i] = change
        #                         change_math += change - y_batch[i]
        #         # 当表内不再更改后跳出循环
        #         if change_math == 0:
        #             break
        #
        #     if t % 5000 == 0:
        #         print(r_batch)
        #         print(y_batch)
        #
        #     # 根据实际Q值表训练神经网络
        #     train_step.run(feed_dict={
        #                     y: y_batch,
        #                     a: a_batch,
        #                     s: s_j_batch
        #     })

        s_t = s_t1
        t += 1

        if t >= 2000 and t % 1000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        state = ""
        if t <= OBSERVE:
            state = "OBSERVE"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("训练次数", t, "/ 训练状态", state,
              "/ 冒险概率", epsilon, "/ 采取行动", action_index, "/ 行动奖励值", r_t)
        print("/ 当前行动Q值 %e" % readout_t[action_index])

        if t % 10000 <= 20:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s: [s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)


def play_game():
    sess = tf.compat.v1.InteractiveSession()
    s, readout, h_fc1 = create_network()
    train_network(s, readout, h_fc1, sess)


def main():
    play_game()
    # action = [1, 1, 0]
    # Game.frame_step(action)


if __name__ == '__main__':
    main()
