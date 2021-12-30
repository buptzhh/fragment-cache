# coding=UTF-8
import gym
from gym import spaces
import numpy as np
from .content_with_fragment import content_with_fragment
from itertools import combinations
import math
import copy
import time

# only for bilibili dataset
MAXSIZE = 204800000
T = 10
test_time = 20
X = 100000
EP_STEP = 1000
request_list = []

block_size = 10240  # content block num = int(content_size / block_size)+1
fragment_freq = 1
content_request_state = {}  # content = {content_id:[time0_req,time1_req,...]}
content_requested_state = {}

request_sum = np.zeros((1, test_time))[0]
requested_sum = np.zeros((1, test_time))[0]

request_sum_size = np.zeros((1, test_time))[0]
requested_sum_size = np.zeros((1, test_time))[0]

cache_init_isFinish = False
cache_size = MAXSIZE
re_size = MAXSIZE
cached_content = {}
start_point = 0
start_time = 0

alpha = 1.01
zipf = [(1 / (i ** alpha)) for i in range(1, 2501)]  # zipf/content.zipf_value为从头请求到每一块的概率


def get_block_num(size):
    return math.ceil(size / block_size)


with open('dataset/bilibili_sort.csv') as f:
    # with open('/data/hdd1/lpm/wiki2018.tr') as f:
    line_count = 0
    for line in f:
        # print(line)
        if line_count == 0:
            line_count += 1
            continue
        info = line.split(',')
        ltime = int(float(info[0]))
        content_id = int(info[1])
        content_size = int(info[2])
        if ltime >= test_time:
            print('read finish')
            break
        if line_count % 100000 == 0:
            print(line_count)
        if 102400 < content_size < 4096000:
            line_count += 1
            content = content_with_fragment(content_id, content_size, ltime)
            request_list.append(content)
            if content_id not in content_request_state:  # 记录历史访问信息
                content_request_state[content_id] = np.zeros((1, test_time))[0]
                content_requested_state[content_id] = np.zeros((1, test_time))[0]
            content_request_state[content_id][ltime] += 1
            request_sum[ltime] += 1
            request_sum_size[ltime] += content_size
            if not cache_init_isFinish:
                content_requested_state[content_id][ltime] += 1
                requested_sum_size[ltime] += content_size
                requested_sum[ltime] += 1
                if content_size <= re_size:  # 初始化
                    if content_id not in cached_content:
                        cached_content[content_id] = content
                        re_size -= content.should_cached_size
                if ltime >= T - 1:
                    start_point = line_count
                    start_time = ltime
                    cache_init_isFinish = True
                    print(len(cached_content))


class CacheSimulator(gym.Env):
    # action_bound = [0, 1, 2, 3, 4]  # 0为不准入，1-5为准入且剔除价值量最少的4种剔除方式
    def __init__(self, name):
        self.name = name
        self.action_space = spaces.Discrete(5)
        self.done = False
        self.observation_space = 5
        self.action_space = 5
        self.cache_size = cache_size
        self.re_size = re_size
        self.cur_point = start_point
        self.cur_time = start_time
        self.action_time = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        # end tabs
        self.step_count = 0
        # value
        self.content_fragment_value = {}
        self.cached_content = {}
        for c in cached_content:
            self.cached_content[c] = copy.deepcopy(cached_content[c])

        self.content_requested_state = copy.deepcopy(content_requested_state)
        self.requested_sum_size = copy.deepcopy(requested_sum_size)
        self.requested_sum = copy.deepcopy(requested_sum)

        self._next_request()
        while request_list[self.cur_point].content_id in self.cached_content:
            self._next_request()
        self._update_value(self.cur_time)
        self.state, self.evict_action = self._get_evict_action()
        print(self.state)
        print(self.evict_action)
        print(self.cur_time)
        pass

    def step(self, action):
        self.action_time[action] += 1
        self.step_count += 1
        if self.step_count % EP_STEP == 0:
            print(self.name, " action:", self.action_time)
            self.done = True
        if action == 0:  # 不准入
            pass
        else:
            if action > len(self.evict_action):
                action = len(self.evict_action)
            for content in self.evict_action[action - 1]:
                self._pop_item(content)
            self._cache_item(request_list[self.cur_point].content_id)

        while True:  # 只有当未命中/命中部分且剩余空间小于请求片段 and evict
            self._next_request()
            if request_list[self.cur_point].content_id in self.cached_content:  # 缓存还有空间和缓存命中的情况
                if self.cached_content[request_list[self.cur_point].content_id].should_cached_size >= request_list[
                    self.cur_point].should_cached_size:
                    continue  # hit
                else:  # hit a part
                    if self.re_size >= request_list[self.cur_point].should_cached_size - self.cached_content[
                        request_list[self.cur_point].content_id].should_cached_size:
                        self._cache_item(request_list[self.cur_point].content_id)
                        continue
            else:  # miss
                if self.re_size >= request_list[self.cur_point].should_cached_size:
                    self._cache_item(request_list[self.cur_point].content_id)
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
        if self.cached_content[_content_id].should_cached_size < 0:
            print('error---', self.cur_point)
            print(self.cached_content[_content_id].should_cached_size, self.cached_content[_content_id].last_fragment_size)
            exit()
        if self.cached_content[_content_id].should_cached_size == 0:
            self.cached_content.pop(_content_id)
            self.content_fragment_value.pop(_content_id)
        else:
            self.update_a_fragment(_content_id)
            self.update_a_value(_content_id)

    def _cache_item(self, _content_id):
        if _content_id not in self.cached_content:
            self.cached_content[_content_id] = copy.deepcopy(request_list[self.cur_point])
            self.re_size -= self.cached_content[_content_id].should_cached_size
        else:
            self.re_size -= request_list[self.cur_point].should_cached_size - self.cached_content[
                _content_id].should_cached_size
            self.cached_content[_content_id].should_cached_size = request_list[self.cur_point].should_cached_size
        # 更新缓存内容的价值
        self.update_a_fragment(_content_id)
        self.update_a_value(_content_id)


    def update_a_fragment(self, _content_id):
        freq = sum(self.content_requested_state[_content_id][self.cur_time - T + 1:self.cur_time + 1])
        fragment_block = 0
        fragment_block_value = 0
        if self.cached_content[_content_id].should_cached_size <= 5 * block_size:
            self.cached_content[_content_id].last_fragment_size = self.cached_content[_content_id].should_cached_size
        for i in list(reversed(range(math.ceil(self.cached_content[_content_id].should_cached_size / block_size)))):
            fragment_block_value += zipf[i] / self.cached_content[_content_id].zipf_value
            fragment_block += 1
            if fragment_block < 5:
                continue
            if fragment_block >= 100 or freq * fragment_block_value >= fragment_freq:
                break
        if self.cached_content[_content_id].should_cached_size <= fragment_block * block_size:
            self.cached_content[_content_id].last_fragment_size = self.cached_content[_content_id].should_cached_size
        self.cached_content[_content_id].last_fragment_size = fragment_block * block_size
        pass

    def update_a_value(self, _content_id):
        freq = sum(self.content_requested_state[_content_id][self.cur_time - T + 1:self.cur_time + 1])
        frag_por = sum(zipf[math.ceil((self.cached_content[_content_id].should_cached_size
                                       - self.cached_content[_content_id].last_fragment_size) / block_size):
                            math.ceil(self.cached_content[_content_id].should_cached_size / block_size)]) / \
                   self.cached_content[_content_id].zipf_value
        self.content_fragment_value[_content_id] = X * freq * frag_por / self.cached_content[
            _content_id].last_fragment_size


    def _next_request(self):
        self.cur_point += 1
        req_id = request_list[self.cur_point].content_id
        _content_size = request_list[self.cur_point].size
        req_size = request_list[self.cur_point].request_size

        req_time = request_list[self.cur_point].time

        if req_time != self.cur_time:
            self.cur_time = req_time
            self._update_value(self.cur_time)
        if req_id not in self.content_requested_state:
            self.content_requested_state[req_id] = np.zeros((1, test_time))[0]

        self.content_requested_state[req_id][req_time] += 1
        self.requested_sum_size[req_time] += req_size

        if req_id in self.cached_content:  # 更新已经缓存内容的价值
            self.update_a_fragment(req_id)
            self.update_a_value(req_id)

    def _get_min_val(self):
        sorted_content = sorted(self.content_fragment_value.items(), key=lambda x: x[1], reverse=False)
        sorted_content = sorted_content[:20]
        min_val = {}
        for content, value in sorted_content:
            min_val[content] = value
        return min_val

    def _update_value(self, _time):
        self.content_fragment_value = {}
        for item in self.cached_content:
            self.update_a_fragment(item)
            self.update_a_value(item)
        delete_list = []
        for item in self.content_requested_state:
            if item not in self.cached_content:
                if sum(self.content_requested_state[item][_time - T + 1:_time + 1]) == 0:
                    delete_list.append(item)
        for item in delete_list:
            self.content_requested_state.pop(item)

    def _r_func(self, _time):  # !!!
        hit_size = 0
        for _content in self.cached_content:  # 通过计算字节命中率得到reward
            hit_size += self.cached_content[_content].should_cached_size * \
                        sum(content_request_state[_content][_time:_time + T]) * \
                        sum(zipf[: math.ceil(
                            self.cached_content[_content].should_cached_size / block_size)]) / content.zipf_value
        total_size = sum(request_sum_size[_time:_time + T]) - self.requested_sum_size[_time]
        return hit_size / total_size

    def _get_evict_action(self):
        min_val = self._get_min_val()
        req_id = request_list[self.cur_point].content_id
        req_time = request_list[self.cur_point].time
        if req_id not in self.cached_content:
            req_size = request_list[self.cur_point].should_cached_size
            req_value = X * sum(self.content_requested_state[req_id][req_time - T + 1:req_time + 1]) * \
                        sum(zipf[:math.ceil(req_size / block_size)]) / request_list[self.cur_point].zipf_value
        else:
            req_size = request_list[self.cur_point].should_cached_size - self.cached_content[
                req_id].should_cached_size

            req_value = X * sum(self.content_requested_state[req_id][req_time - T + 1:req_time + 1]) * \
                        sum(zipf[math.ceil(self.cached_content[req_id].should_cached_size / block_size)
                                 :math.ceil(request_list[self.cur_point].should_cached_size / block_size)]) / \
                        request_list[self.cur_point].zipf_value
        req_value = req_value / req_size

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
                              (0, 4 - len(evict_candidators))), \
                       [candidators[k] for k, v in evict_candidators]  # state, action
            for _content in contents:
                if min_val[_content] == 0:
                    self._pop_item(_content)
            return self._get_evict_action()
        return np.pad(np.array([0] + [req_value - v for k, v in evict_candidators]), (0, 4 - len(evict_candidators))), \
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
