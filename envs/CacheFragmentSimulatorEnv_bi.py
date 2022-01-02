# coding=UTF-8
import gym
from gym import spaces
import numpy as np
from .content_with_fragment import content_with_fragment
from itertools import combinations
import math
import copy
import time
from collections import OrderedDict, defaultdict

# only for bilibili dataset
MAXSIZE = 2048000000
T = 60
Fu_window = 10
test_time = 3600
X = 1000000
EP_STEP = 1000

block_size = 10240  # content block num = int(content_size / block_size)+1
fragment_freq = 1

alpha = 1.01
zipf = [(1 / (i ** alpha)) for i in range(1, 2501)]  # zipf/content.zipf_value为从头请求到每一块的概率


def get_block_num(size):
    return math.ceil(size / block_size)


class CacheSimulator(gym.Env):
    # action_bound = [0, 1, 2, 3, 4]  # 0为不准入，1-5为准入且剔除价值量最少的4种剔除方式
    def __init__(self, name):
        self.name = name
        self.action_space = spaces.Discrete(5)
        self.done = False
        self.observation_space = 5
        self.action_space = 5
        self.infilepath = 'dataset/122.228.93.41.csv'
        self.infile = open(self.infilepath)
        self.infile_for_future = open(self.infilepath)
        self.infile.readline()
        self.infile_for_future.readline()
        self.future_total_size = np.zeros((1, Fu_window))[0]
        self.cache_init_isFinish = False
        self.observation_space = 5
        self.action_space = 5
        self.cache_size = MAXSIZE
        self.re_size = MAXSIZE
        self.cur_point = 0
        self.cur_time = 0
        self.action_time = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        # end tabs
        self.line_count = 0
        # value
        self.content_fragment_value = {}
        self.cached_content = {}
        self.content_requested_state = {}
        self.delay_req = {}

        self.cur_content = None
        self.cur_time_pos = 0

        self.low_value_pool = OrderedDict()
        self.mid_value_pool = OrderedDict()

        self.state = None
        self.evict_action = None
        self.init_cache()
        pass

    def read_a_line(self):
        line = self.infile.readline()
        info = line.split(',')
        ltime = int(float(info[0]))
        content_id = int(info[1])
        content_size = int(info[2])
        if ltime >= test_time:
            print('dataset is too short')

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
        time_pos = start_time % Fu_window
        _cur_time = start_time
        self.future_total_size[time_pos] = 0
        delete_list = []
        for content_id in self.delay_req:
            self.delay_req[content_id][time_pos] = 0
            if sum(self.delay_req[content_id]) == 0:
                delete_list.append(content_id)
        for di in delete_list:
            self.delay_req.pop(di)
        while start_time == _cur_time:
            line = self.infile_for_future.readline()
            info = line.split(',')
            _cur_time = int(float(info[0]))
            content_id = int(info[1])
            self.future_total_size[time_pos] += int(info[2])
            if content_id in self.delay_req:
                self.delay_req[content_id][time_pos] += 1
            else:
                self.delay_req[content_id] = np.zeros((1, Fu_window))[0]
                self.delay_req[content_id][time_pos] = 1

    def init_cache(self):
        while not self.cache_init_isFinish:
            self.cur_content = self.read_a_line()
            self.cur_time_pos = self.cur_content.time % T
            if self.cur_content.content_id not in self.content_requested_state:  # 记录历史访问信息
                self.content_requested_state[self.cur_content.content_id] = np.zeros((1, T))[0]
            self.content_requested_state[self.cur_content.content_id][self.cur_time_pos] += 1
            # if self.cur_content.time >= T - 1:  # 第九秒开始初始缓存内容
            if self.cur_content.size <= self.re_size:  # 初始化
                if self.cur_content.content_id not in self.cached_content:
                    self.cached_content[self.cur_content.content_id] = copy.deepcopy(self.cur_content)
                    self.cached_content[self.cur_content.content_id].should_cached_size = self.cur_content.size - (
                                self.cur_content.size % block_size)
                    self.re_size -= self.cached_content[self.cur_content.content_id].should_cached_size
            else:
                self.cache_init_isFinish = True
                print(len(self.cached_content))
                break
        for i in range(self.cur_content.time + 1 + Fu_window):  # 提前读了第十秒的内容
            self.read_future()

        self._update_value()
        self._next_request()
        self.step(0)
        print(self.state)
        print(self.evict_action)
        print(self.cur_time)

    def step(self, action):
        self.action_time[action] += 1
        self.line_count += 1
        if self.line_count % EP_STEP == 0:
            print(self.name, " action:", self.action_time)
            self.done = True
        if action == 0:  # 不准入
            pass
        else:
            if action > len(self.evict_action):
                action = len(self.evict_action)
            for content in self.evict_action[action - 1]:
                self._pop_item(content)
            self._cache_item(self.cur_content.content_id)

        while True:  # 只有当未命中/命中部分且剩余空间小于请求片段 and evict
            self._next_request()
            if self.cur_content.content_id in self.cached_content:  # 缓存还有空间和缓存命中的情况
                if self.cached_content[
                    self.cur_content.content_id].should_cached_size >= self.cur_content.should_cached_size:
                    continue  # hit
                else:  # hit a part
                    if self.re_size >= self.cur_content.should_cached_size - self.cached_content[
                        self.cur_content.content_id].should_cached_size:
                        self._cache_item(self.cur_content.content_id)
                        continue
            else:  # miss
                if self.re_size >= self.cur_content.should_cached_size:
                    self._cache_item(self.cur_content.content_id)
                    continue
            self.state, self.evict_action = self._get_evict_action()
            if len(self.evict_action) != 0:
                break
        reward = self._r_func(self.cur_time)
        # print(self.state)
        # print(self.evict_action)
        return self.state, reward, self.done

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
            freq += self.delay_req[_content_id][(self.cur_time + 1) % Fu_window]
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
        req_time = self.cur_content.time
        if req_time != self.cur_time:
            self.cur_time = req_time
            self._update_value()
            self.read_future()
            self._update_value()
        if req_id not in self.content_requested_state:
            self.content_requested_state[req_id] = np.zeros((1, T))[0]

        self.content_requested_state[req_id][self.cur_time_pos] += 1

        if req_id in self.cached_content:  # 更新已经缓存内容的价值
            self.update_a_fragment(req_id)
            self.update_a_value(req_id)

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
            if item not in self.cached_content:
                if sum(self.content_requested_state[item]) == 0:
                    delete_list.append(item)
            else:
                self.update_a_fragment(item)
                self.update_a_value(item)
        for item in delete_list:
            self.content_requested_state.pop(item)
        self.update_value_pool()

    def update_value_pool(self):
        # print(self.content_fragment_value)
        sorted_content = sorted(self.content_fragment_value.items(), key=lambda x: x[1], reverse=False)

        if len(sorted_content) > 100:
            self.low_value_pool = OrderedDict(sorted_content[:100])
            self.mid_value_pool = OrderedDict(sorted_content[100:])
        else:
            self.low_value_pool = OrderedDict(sorted_content)
            self.mid_value_pool = OrderedDict()

    def _r_func(self, _time):  # !!!
        hit_size = 0
        for content in self.cached_content:  # 通过计算字节命中率得到reward,结果会稍微和实际有区别
            if content in self.delay_req:
                hit_size += sum(self.delay_req[content]) * self.cached_content[content].should_cached_size
        total_time = sum(self.future_total_size)
        return hit_size / total_time

    def _get_evict_action(self):
        min_val = self._get_min_val()
        req_id = self.cur_content.content_id
        req_freq = sum(self.content_requested_state[req_id])
        if req_id in self.delay_req:
            req_freq += self.delay_req[req_id][(self.cur_time + 1) % Fu_window]
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
                       20):  # 优化方案：https://leetcode-cn.com/problems/combination-sum-ii/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-3/
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
        return np.pad(np.array([0] + [req_value - v for k, v in evict_candidators]), (0, 4 - len(evict_candidators)),
                      'constant'), \
               [candidators[k] for k, v in evict_candidators]  # state, action

    def reset(self):
        print(self.name, " reset, cache num:", len(self.cached_content), " re_size:", self.re_size, " cur time:",
              self.cur_time, " cur point： ", self.cur_point, time.asctime(time.localtime(time.time())))
        self.done = False
        return self.state

    def render(self, mode=''):
        return None

    def close(self):
        return None

    def combinaSum(self):
        pass
