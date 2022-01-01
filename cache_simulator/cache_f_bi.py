import gym
from gym import spaces
import numpy as np
from cahce.envs.content_with_fragment import content_with_fragment
from cahce.cahce_algorithm.lru import LRUCache
from cahce.cahce_algorithm.lfu import LFUCache
from itertools import combinations
import multiprocessing
import threading
import tensorflow as tf
import gym
import random
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import pandas as pd
import math
import copy
import time
from collections import OrderedDict, defaultdict

# only for bilibili dataset
MAXSIZE = 2048000000
T = 60
test_time = 3660
X = 100000  # values param
EP_STEP = 1000

block_size = 10240  # content block num = math.ceil(content_size / block_size)
fragment_freq = 1

alpha = 1.01
zipf = [(1 / (i ** alpha)) for i in range(1, 2501)]  # zipf/content.zipf_value为从头请求到每一块的概率

# Network param
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 250
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
N_S = 5
N_A = 5
gtotal_step = 0
lru = LRUCache(MAXSIZE, int(test_time))
lfu = LFUCache(MAXSIZE, int(test_time))


def get_block_num(size):
    return int(math.ceil(size / block_size))


class ACNet(object):

    def __init__(self, scope, globalAC=None):
        if scope == 'Global_Net':  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                # print(self.s)
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)  # min a_loss = max exp_v

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def choose_action(self, s):  # run by a local
        # print(s[np.newaxis, :])
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        # max_index = 0
        # list_index = 0
        # for num in prob_weights[0]:
        #     if num > prob_weights[0][max_index]:
        #         max_index = list_index
        #     list_index += 1
        # action = max_index
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action



class Simulator(gym.Env):
    # action_bound = [0, 1, 2, 3, 4]  # 0为不准入，1-5为准入且剔除价值量最少的4种剔除方式
    def __init__(self, name):
        self.name = name
        self.done = False
        self.cache_init_isFinish = False
        self.observation_space = 5
        self.action_space = 5

        self.infilepath = '../dataset/122.228.93.41.csv'
        self.infile = open(self.infilepath)
        self.infile_for_future = open(self.infilepath)
        self.infile.readline()
        self.infile_for_future.readline()

        self.cache_size = MAXSIZE
        self.re_size = MAXSIZE
        self.cur_point = 0
        self.cur_time = 0
        self.action_time = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        # end tabs
        self.step_count = 0
        # value
        self.content_fragment_value = {}
        self.cached_content = {}
        self.content_requested_state = {}
        self.delay_req = {}

        self.cur_content = None
        self.cur_time_pos = 0

        self.re_size_sum = 0
        self.hit_time = np.zeros((1, int(test_time)))[0]
        self.total_time = np.zeros((1, int(test_time)))[0]
        self.hit_size = np.zeros((1, int(test_time)))[0]
        self.total_size = np.zeros((1, int(test_time)))[0]
        self.back_size = np.zeros((1, int(test_time)))[0]
        self.low_value_pool = OrderedDict()
        self.mid_value_pool = OrderedDict()

        self.state = None
        self.evict_action = None
        self.init_cache()
        pass

    def init_cache(self):
        while not self.cache_init_isFinish:
            self.cur_content = self.read_a_line()
            self.cur_time_pos = self.cur_content.time % T
            if self.cur_content.content_id not in self.content_requested_state:  # 记录历史访问信息
                self.content_requested_state[self.cur_content.content_id] = np.zeros((1, T))[0]
            self.content_requested_state[self.cur_content.content_id][self.cur_time_pos] += 1
            # if self.cur_content.time >= T - 1:  # 第九秒开始初始缓存内容
            if self.cur_content.size <= self.re_size:  # 初始化
                req_history = sum(self.content_requested_state[self.cur_content.content_id])
                lru.get(self.cur_content.content_id)
                lfu.get(self.cur_content.content_id, req_history)
                if self.cur_content.content_id not in self.cached_content:
                    lru.put(self.cur_content.content_id, self.cur_content.size)
                    lfu.put(self.cur_content.content_id, self.cur_content.size, req_history)
                    self.cached_content[self.cur_content.content_id] = copy.deepcopy(self.cur_content)
                    self.cached_content[self.cur_content.content_id].should_cached_size = self.cur_content.size - (self.cur_content.size % block_size)
                    self.re_size -= self.cached_content[self.cur_content.content_id].should_cached_size
            else:
                self.cache_init_isFinish = True
                print(len(self.cached_content))
                break
        for i in range(self.cur_content.time + 2):  # 提前读了下一秒的内容
            self.read_future()

        self._update_value()
        self._next_request()
        self.step(0)
        print(self.state)
        print(self.evict_action)
        print(self.cur_time)

    def step(self, action):
        self.action_time[action] += 1
        self.step_count += 1
        if self.step_count % EP_STEP == 0:
            print(self.name, " action:", self.action_time, "cached num", len(self.cached_content), " re_size:",
                  self.re_size, " cur time:",
                  self.cur_time, " cur point： ", self.cur_point, time.asctime(time.localtime(time.time())))
        if action == 0:  # 不准入
            pass
        else:
            if action > len(self.evict_action):
                action = len(self.evict_action)
            for content in self.evict_action[action - 1]:
                self._pop_item(content)
            self._cache_item(self.cur_content.content_id)

        while True:  # 只有当未命中/命中部分且剩余空间小于请求片段 and evict
            if self._next_request():
                return [], self.done
            if self.cur_content.content_id in self.cached_content:  # 缓存还有空间和缓存命中的情况
                if self.cached_content[self.cur_content.content_id].should_cached_size >= self.cur_content.should_cached_size:
                    continue  # hit
                else:  # hit a part
                    if self.re_size >= self.cur_content.should_cached_size - self.cached_content[self.cur_content.content_id].should_cached_size:
                        self._cache_item(self.cur_content.content_id)
                        continue
            else:  # miss
                if self.re_size >= self.cur_content.should_cached_size:
                    self._cache_item(self.cur_content.content_id)
                    continue
            self.state, self.evict_action = self._get_evict_action()
            if len(self.evict_action) != 0:
                break
        return self.state, self.done

    def _pop_item(self, _content_id):
        self.re_size += self.cached_content[_content_id].last_fragment_size
        self.cached_content[_content_id].should_cached_size -= self.cached_content[_content_id].last_fragment_size
        if self.cached_content[_content_id].should_cached_size == 0:
            self.cached_content.pop(_content_id)
            self.content_fragment_value.pop(_content_id)
            if _content_id in self.low_value_pool:
                self.low_value_pool.pop(_content_id)
        else:
            self.update_a_fragment(_content_id)
            self.update_a_value(_content_id)

    def _cache_item(self, _content_id):
        if _content_id not in self.cached_content:
            self.cached_content[_content_id] = copy.deepcopy(self.cur_content)
            self.re_size -= self.cached_content[_content_id].should_cached_size
        else:
            self.re_size -= self.cur_content.should_cached_size - self.cached_content[_content_id].should_cached_size
            self.cached_content[_content_id].should_cached_size = self.cur_content.should_cached_size
        # 更新缓存内容的价值
        self.update_a_fragment(_content_id)
        self.update_a_value(_content_id)

    def update_a_fragment(self, _content_id):
        freq = sum(self.content_requested_state[_content_id])
        fragment_block = 0
        fragment_block_value = 0
        if self.cached_content[_content_id].should_cached_size <= 5 * block_size:
            self.cached_content[_content_id].last_fragment_size = self.cached_content[_content_id].should_cached_size
        for i in list(
                reversed(range(int(math.ceil(self.cached_content[_content_id].should_cached_size / block_size))))):
            fragment_block_value += zipf[i] / self.cached_content[_content_id].zipf_value
            fragment_block += 1
            if fragment_block < 5:
                continue
            if freq * fragment_block_value >= fragment_freq or fragment_block >= 100:
                break
        if self.cached_content[_content_id].should_cached_size <= fragment_block * block_size:
            self.cached_content[_content_id].last_fragment_size = self.cached_content[_content_id].should_cached_size
        else:
            self.cached_content[_content_id].last_fragment_size = fragment_block * block_size
        pass

    def update_a_value(self, _content_id):
        freq = sum(self.content_requested_state[_content_id])
        if _content_id in self.delay_req:
            freq += self.delay_req[_content_id]
        frag_por = sum(zipf[int(math.ceil((self.cached_content[_content_id].should_cached_size - self.cached_content[
            _content_id].last_fragment_size) / block_size)):
                            int(math.ceil(self.cached_content[_content_id].should_cached_size / block_size))]) / \
                   self.cached_content[_content_id].zipf_value
        self.content_fragment_value[_content_id] = X * freq * frag_por / self.cached_content[_content_id].last_fragment_size

        if _content_id in self.low_value_pool:
            self.low_value_pool.pop(_content_id)
        self.mid_value_pool[_content_id] = self.content_fragment_value[_content_id]
        pass

    def _next_request(self):
        self.cur_point += 1
        self.cur_content = self.read_a_line()

        req_id = self.cur_content.content_id
        _content_size = self.cur_content.size
        req_size = self.cur_content.request_size
        req_time = self.cur_content.time
        if req_time - self.cur_time > 1:
            print('dataset error')
            exit()
        if req_time >= test_time:
            self.done = True
            return True
        # self.re_size_sum += self.re_size / MAXSIZE

        if req_time != self.cur_time:
            self.cur_time = req_time
            self.cur_time_pos = self.cur_time % T
            self.read_future()
            self._update_value()
        if req_id not in self.content_requested_state:
            self.content_requested_state[req_id] = np.zeros((1, T))[0]

        self.content_requested_state[req_id][self.cur_time_pos] += 1
        # self.requested_sum_size[req_time] += req_size

        if req_id in self.cached_content:  # 更新已经缓存内容的价值
            self.update_a_fragment(req_id)
            self.update_a_value(req_id)

            self.hit_time[req_time] += 1
            if self.cached_content[req_id].should_cached_size >= req_size:
                self.hit_size[req_time] += req_size
            else:
                self.hit_size[req_time] += self.cached_content[req_id].should_cached_size
                self.back_size[req_time] += self.cur_content.should_cached_size - self.cached_content[
                    req_id].should_cached_size
        else:
            self.back_size[req_time] += self.cur_content.should_cached_size

        self.total_time[req_time] += 1
        self.total_size[req_time] += req_size

        lru.run(req_id, _content_size, req_time, req_size)
        req_history = sum(self.content_requested_state[req_id])
        lfu.run(req_id, _content_size, req_time, req_size, req_history)
        return False

    def read_a_line(self):
        line = self.infile.readline()
        info = line.split(',')
        ltime = int(float(info[0]))
        content_id = int(info[1])
        content_size = int(info[2])
        if ltime >= test_time:
            print('read finish')
        if 102400 < content_size < 4096000:
            cur_content = content_with_fragment(content_id, content_size, ltime)
            return cur_content
        else:
            return self.read_a_line()

    def read_future(self):
        self.delay_req.clear()
        line = self.infile_for_future.readline()
        info = line.split(',')
        start_time = int(float(info[0]))
        _cur_time = start_time
        while start_time == _cur_time:
            line = self.infile_for_future.readline()
            info = line.split(',')
            _cur_time = int(float(info[0]))
            content_id = int(info[1])
            if content_id in self.delay_req:
                self.delay_req[content_id] += 1
            else:
                self.delay_req[content_id] = 1
        # _cur_time_pos = _cur_time % time_slot
        # while start_time == _cur_time:
        #     line = self.infile_for_future.readline()
        #     info = line.split(',')
        #     _cur_time = int(float(info[0]))
        #     content_id = int(info[1])
        #     if content_id in delay_req:
        #         delay_req[content_id][_cur_time_pos] += 1
        #     else:
        #         delay_req[content_id] = np.zeros((1, time_slot))[0]
        #         delay_req[content_id][_cur_time_pos] += 1
        # _cur_time_pos = _cur_time % time_slot
        # for item in delay_req:
        #     delay_req[item][_cur_time_pos] = 0
        #     if sum(delay_req[item]) == 0:
        #         delay_req.pop(item)

    def _get_min_val(self):
        if len(self.low_value_pool) < 21:
            self.update_value_pool()
        min_list = list(self.low_value_pool.items())
        min_val = {}
        for content, value in min_list[:20]:
            min_val[content] = value
        if self.cur_content.content_id in min_val:
            min_val.pop(self.cur_content.content_id)
            min_val[min_list[20][0]] = min_list[20][1]
        return min_val

    def _update_value(self):
        self.content_fragment_value.clear()
        delete_list = []
        for item in self.content_requested_state:
            self.content_requested_state[item][self.cur_time_pos] = 0
            req_history = sum(self.content_requested_state[item])
            lfu.get(item, req_history)
            if item not in self.cached_content:
                if req_history == 0:
                    delete_list.append(item)
            else:
                self.update_a_fragment(item)
                self.update_a_value(item)
        for item in delete_list:
            self.content_requested_state.pop(item)
        self.update_value_pool()

    def update_value_pool(self):
        sorted_content = sorted(self.content_fragment_value.items(), key=lambda x: x[1], reverse=False)
        if len(sorted_content) > 100:
            self.low_value_pool = OrderedDict(sorted_content[:100])
            self.mid_value_pool = OrderedDict(sorted_content[100:])
        else:
            self.low_value_pool = OrderedDict(sorted_content)
            self.mid_value_pool = OrderedDict()


    def _get_evict_action(self):
        min_val = self._get_min_val()
        req_id = self.cur_content.content_id
        req_freq = sum(self.content_requested_state[req_id])
        if req_id in self.delay_req:
            req_freq += self.delay_req[req_id]
        if req_id not in self.cached_content:
            req_size = self.cur_content.should_cached_size
            req_value = X * req_freq * sum(zipf[:int(math.ceil(req_size / block_size))]) / (
                        self.cur_content.zipf_value * req_size)
        else:
            req_size = self.cur_content.should_cached_size - self.cached_content[req_id].should_cached_size
            req_value = X * req_freq * sum(
                zipf[int(math.ceil(self.cached_content[req_id].should_cached_size / block_size))
                     :int(math.ceil(self.cur_content.should_cached_size / block_size))]) / (
                                    self.cur_content.zipf_value * req_size)

        contents = [key for key in min_val.keys()]
        evict_candidators = {}
        candidators = {}
        candidators_number = 0
        for i in range(1,
                       20):
            for combine in combinations(contents, i):
                evict_candidator = list(combine)
                total_size = 0
                total_value = 0
                for item in evict_candidator:
                    total_size += self.cached_content[item].last_fragment_size
                    total_value += min_val[item]
                if total_size + self.re_size >= req_size:
                    candidators_number += 1
                    evict_candidators[candidators_number] = total_value
                    candidators[candidators_number] = evict_candidator
            if len(evict_candidators) >= 4:
                break
        evict_candidators = sorted(evict_candidators.items(), key=lambda x: x[1], reverse=False)
        evict_candidators = evict_candidators[:4]
        if len(evict_candidators) < 4:
            if 0 not in list(min_val.values()):
                return np.pad(np.array([0] + [req_value - v for k, v in evict_candidators]),
                              (0, 4 - len(evict_candidators)), 'constant'), \
                       [candidators[k] for k, v in evict_candidators]  # state, action
            for _content in contents:
                if min_val[_content] == 0:
                    self._pop_item(_content)
            return self._get_evict_action()
        return np.pad(np.array([0] + [req_value - v for k, v in evict_candidators]), (0, 4 - len(evict_candidators)), 'constant'), \
               [candidators[k] for k, v in evict_candidators]  # state, action


def draw(z1, z2, z3, y1, y2, y3, t1, t2, t3):
    # sns.set(color_codes=True)
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    X = np.arange(len(z1))

    fig = plt.figure()

    # ----------------------------------------
    ax1 = fig.add_subplot(1, 3, 1)
    # 绘制折线图
    ax1.plot(X, z1, marker='o', mec='r', mfc='w', label="OUR", alpha=0.5)
    ax1.plot(X, z2, marker='o', mec='b', mfc='w', label="LRU", alpha=0.5)
    ax1.plot(X, z3, marker='o', mec='y', mfc='w', label="LFU", alpha=0.5)

    ax1.set_xlabel("time")
    ax1.set_ylabel("hit rate(%)")
    ax1.set_title('Hit Rate Of Each Cache Algorithm')

    # ----------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    # 绘制折线图
    ax2.plot(X, y1, marker='o', mec='r', mfc='w', label="OUR", alpha=0.5)
    ax2.plot(X, y2, marker='o', mec='b', mfc='w', label="LRU", alpha=0.5)
    ax2.plot(X, y3, marker='o', mec='y', mfc='w', label="LFU", alpha=0.5)

    ax2.set_xlabel("time")
    ax2.set_ylabel("traffic(%)")
    ax2.set_title('Back to source Of Each Cache Algorithm')
    # ----------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    # 绘制折线图
    ax3.plot(X, t1, marker='o', mec='r', mfc='w', label="OUR", alpha=0.5)
    ax3.plot(X, t2, marker='o', mec='b', mfc='w', label="LRU", alpha=0.5)
    ax3.plot(X, t3, marker='o', mec='y', mfc='w', label="LFU", alpha=0.5)

    ax3.set_xlabel("time")
    ax3.set_ylabel("byte hit rate(%)")
    ax3.set_title('Byte hit rate Of Each Cache Algorithm')
    # 显示图例
    plt.legend()
    plt.savefig('result_f_bi2.png')
    plt.show()


if __name__ == "__main__":
    simulator = Simulator("worker")
    state = simulator.state

    SESS = tf.Session()
    with tf.device("/cpu:0"):
        GLOBAL_AC = ACNet('Global_Net')
        saver = tf.train.Saver()
        # saver  =tf.train.import_meta_graph("model/cache_model_s1.ckpt.meta")
        saver.restore(SESS, "../model/fragment_model_s1/cache_model_bi.ckpt")  # 载入参数，参数保存在两个文件中，不过restore会自己寻找
        # graph = tf.get_default_graph()
        # S = graph.get_operation_by_name('S').outputs[0]
        done = False
        while not done:
            # action = SESS.run(state, feed_dict={S,})
            action = GLOBAL_AC.choose_action(state)
            state, done = simulator.step(action)
    print(simulator.action_time)
    ori_time = 60
    filename = 'bi_f_result.txt'
    with open(filename, 'a') as file_object:
        total_time = sum([simulator.total_time[i] for i in range(ori_time, test_time)])
        total_size = sum([simulator.total_size[i] for i in range(ori_time, test_time)])
        file_object.write("\n\ncache size :" + str(MAXSIZE) + " T:" + str(T)
                          + "\nOURS hit rate:" + str(
            sum([simulator.hit_time[i] for i in range(ori_time, test_time)]) / total_time) + " "
                          + str([simulator.hit_time[i] / simulator.total_time[i] for i in range(ori_time, test_time)])
                          + "\nLRU hit rate:" + str(
            sum([lru.hit_time[i] for i in range(ori_time, test_time)]) / total_time) + " "
                          + str([lru.hit_time[i] / lru.total_time[i] for i in range(ori_time, test_time)])
                          + "\nLFU hit rate:" + str(
            sum([lfu.hit_time[i] for i in range(ori_time, test_time)]) / total_time) + " "
                          + str([lfu.hit_time[i] / lfu.total_time[i] for i in range(ori_time, test_time)])
                          + "\nOUR back size:"
                          + str([simulator.back_size[i] for i in range(ori_time, test_time)])
                          + "\nLRU back size:"
                          + str([lru.back_size[i] for i in range(ori_time, test_time)])
                          + "\nLFU back size:"
                          + str([lfu.back_size[i] for i in range(ori_time, test_time)])
                          + "\nORU byte hit rate:" + str(
            sum([simulator.hit_size[i] for i in range(ori_time, test_time)]) / total_size) + " "
                          + str([simulator.hit_size[i] / simulator.total_size[i] for i in range(ori_time, test_time)])
                          + "\nLRU byte hit rate:" + str(
            sum([lru.hit_size[i] for i in range(ori_time, test_time)]) / total_size) + " "
                          + str([lru.hit_size[i] / lru.total_size[i] for i in range(ori_time, test_time)])
                          + "\nLFU byte hit rate:" + str(
            sum([lfu.hit_size[i] for i in range(ori_time, test_time)]) / total_size) + " "
                          + str([lfu.hit_size[i] / lfu.total_size[i] for i in range(ori_time, test_time)])
                          + "\ntotal times:" + str(total_time) + str(
            [simulator.total_time[i] for i in range(ori_time, test_time)])
                          + "\ntotal size:" + str(total_size) + str(
            [simulator.total_size[i] for i in range(ori_time, test_time)]))

    draw([simulator.hit_time[i] / simulator.total_time[i] for i in range(T, test_time)],
         [lru.hit_time[i] / lru.total_time[i] for i in range(T, test_time)],
         [lfu.hit_time[i] / lfu.total_time[i] for i in range(T, test_time)],
         [simulator.back_size[i] for i in range(T, test_time)],
         [lru.back_size[i] for i in range(T, test_time)],
         [lfu.back_size[i] for i in range(T, test_time)],
         [simulator.hit_size[i] / simulator.total_size[i] for i in range(T, test_time)],
         [lru.hit_size[i] / lru.total_size[i] for i in range(T, test_time)],
         [lfu.hit_size[i] / lfu.total_size[i] for i in range(T, test_time)],
         )
    # 0.file_object.write(330958693712246
    # 0.09387754945803417
