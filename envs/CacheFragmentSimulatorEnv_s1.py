import gym
from gym import spaces
import numpy as np
from .content_with_fragment import content_with_fragment
from itertools import combinations
import math
import copy
# only for s1dataset
MAXSIZE = 104857600
T = 40
test_time = 7200
X = 100000
EP_STEP = 500
request_list = []

block_size = 2048  # content block num = int(content_size / block_size)+1
fragment_freq = 1000
block_num = 512
content_request_state = {}  # content = {content_id:[time0_req,time1_req,...]}
content_requested_state = {}

request_sum_size = np.zeros((1, test_time))[0]
requested_sum_size = np.zeros((1, test_time))[0]

cache_init_isFinish = False
cache_size = MAXSIZE
re_size = MAXSIZE

cahced_content = {}  # content = {content_id:content()}
start_point = 0
start_time = 0


def add_content_requested_state(_content_requested_state, _time, _req_size):
    add_block = math.ceil(_req_size / block_size)
    for i in range(add_block):
        _content_requested_state[i][_time] += 1


def get_block_num(size):
    return math.ceil(size / block_size)


with open('dataset/s1.csv') as f:
    # with open('/data/hdd1/lpm/wiki2018.tr') as f:
    line_count = 0
    for line in f:
        info = line.split(',')
        time = int(float(info[0]))
        content_id = int(info[1])
        content_size = int(info[2])
        if line_count % 100000 == 0:
            print(line_count)
        line_count += 1

        content = content_with_fragment(content_id, content_size, time)
        content.init()

        request_list.append(content)
        if content_id not in content_request_state:
            content_request_state[content_id] = [np.zeros((1, test_time), dtype=int)[0] for i in range(get_block_num(content_size))]
            content_requested_state[content_id] = [np.zeros((1, test_time), dtype=int)[0] for i in range(get_block_num(content_size))]
        add_content_requested_state(_content_requested_state=content_request_state[content_id], _time=time,
                                    _req_size=content.request_size)
        # content_request_state[content_id][ltime] += 1
        request_sum_size[time] += content.request_size
        if not cache_init_isFinish:
            add_content_requested_state(_content_requested_state=content_requested_state[content_id], _time=time,
                                        _req_size=content.request_size)
            # content_requested_state[content_id][ltime] += 1
            requested_sum_size[time] += content.request_size

            if content_size <= re_size:  # 初始化
                if content_id not in cahced_content:
                    cahced_content[content_id] = content
                    re_size -= content.should_cached_size

            if time >= T:
                start_point = line_count
                start_time = time
                cache_init_isFinish = True
                print(len(cahced_content))
    print('read finish')


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
        self.line_count = 0
        # value
        self.content_fragment_value = {}
        self.cached_content = {}
        for c in cahced_content:
            self.cached_content[c] = copy.deepcopy(cahced_content[c])

        self.content_requested_state = copy.deepcopy(content_requested_state)
        self.requested_sum_size = copy.deepcopy(requested_sum_size)
        # self.requested_sum = copy.deepcopy(requested_sum)

        # self._next_request()
        # while request_list[self.cur_point].content_id in self.cached_content:
        #     self._next_request()
        self._update_value(self.cur_time)
        self.state, self.evitc_action = self._get_evict_action()
        print(self.state)
        print(self.evitc_action)
        print(self.cur_time)
        pass

    def step(self, action):
        self.action_time[action] += 1
        self.line_count += 1
        if self.line_count % EP_STEP == 0:
            print(self.name, " action:", self.action_time)
            self.done = True
        if action == 0:  # 不准入
            pass
        else:
            if action > len(self.evitc_action):
                action = len(self.evitc_action)
            for content in self.evitc_action[action - 1]:
                self._pop_item(content)
            self._cache_item(request_list[self.cur_point].content_id)

        self._next_request()
        self.state, self.evitc_action = self._get_evict_action()
        while True:
            if request_list[self.cur_point].content_id in self.cached_content:  # 缓存还有空间和缓存命中的情况
                if self.cached_content[request_list[self.cur_point].content_id].should_cached_size >= request_list[
                    self.cur_point].request_size:
                    pass
                else:
                    if self.re_size > request_list[self.cur_point].should_cached_size - self.cached_content[
                        request_list[self.cur_point].content_id].should_cached_size:
                        self._cache_item(request_list[self.cur_point].content_id)
                    elif len(self.evitc_action) != 0:
                        break
            else:
                if self.re_size > request_list[self.cur_point].should_cached_size:
                    self._cache_item(request_list[self.cur_point].content_id)
                elif len(self.evitc_action) != 0:
                    break
            self._next_request()
            self.state, self.evitc_action = self._get_evict_action()
        reward = self._r_func(self.cur_time)
        print(self.state)
        print(self.evitc_action)
        return self.state, reward, self.done

    def _pop_item(self, _content_id):
        self.re_size += self.cached_content[_content_id].last_fragment_size
        self.cached_content[_content_id].should_cached_size -= self.cached_content[_content_id].last_fragment_size
        self.update_a_fragment(_content_id)
        if self.cached_content[_content_id].last_fragment_size == 0:
            self.cached_content.pop(_content_id)
            self.content_fragment_value.pop(_content_id)
        else:
            self.update_a_value(_content_id)

    def _cache_item(self, _content_id):
        if _content_id not in self.cached_content:
            self.cached_content[_content_id] = request_list[self.cur_point]
            self.re_size -= self.cached_content[_content_id].should_cached_size
        else:
            self.re_size -= request_list[self.cur_point].should_cached_size - self.cached_content[
                _content_id].should_cached_size
            self.cached_content[_content_id].should_cached_size = request_list[self.cur_point].should_cached_size
            self.update_a_fragment(_content_id)
        # 更新缓存内容的价值
        self.update_a_value(_content_id)

    def update_a_fragment(self, _content_id):
        freq = 0
        fragment_block = 0
        for i in list(reversed(range(math.ceil(self.cached_content[_content_id].should_cached_size / block_size)))):
            freq += sum(self.content_requested_state[_content_id][i][self.cur_time - T + 1:self.cur_time + 1])
            fragment_block += 1
            if freq >= fragment_freq:
                break
        self.cached_content[_content_id].last_fragment_size = fragment_block * block_size
        pass

    def update_a_value(self, _content_id):
        sum_fre = 0
        for i in range(math.ceil((self.cached_content[_content_id].should_cached_size -
                                  self.cached_content[_content_id].last_fragment_size) / block_size),
                       math.ceil(self.cached_content[_content_id].should_cached_size / block_size)):
            sum_fre += sum(self.content_requested_state[_content_id][i][self.cur_time - T + 1:self.cur_time + 1])
        self.content_fragment_value[_content_id] = X * sum_fre / self.cached_content[_content_id].last_fragment_size

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
            self.content_requested_state[req_id] = [np.zeros((1, test_time), dtype=int)[0] for i in range(get_block_num(content_size))]

        add_content_requested_state(_content_requested_state=self.content_requested_state[content_id], _time=time,
                                    _req_size=content.request_size)
        if req_id in self.cached_content:  # 更新已经缓存内容的价值
            self.update_a_value(req_id)
        # self.content_requested_state[req_id][req_time] += 1
        self.requested_sum_size[req_time] += req_size

    def _get_min_val(self):
        sorted_content = sorted(self.content_fragment_value.items(), key=lambda x: x[1], reverse=False)
        sorted_content = sorted_content[:100]
        min_val = {}
        for content, value in sorted_content:
            min_val[content] = value
        return min_val

    def _update_value(self, time):
        self.content_fragment_value = {}
        for item in self.cached_content:
            self.update_a_value(item)
        delete_list = []
        for item in self.content_requested_state:
            if item not in self.cached_content:
                sum_fre = 0
                for i in range(block_num):
                    sum_fre += sum(self.content_requested_state[item][i][time - T + 1:time + 1])
                if sum_fre == 0:
                    delete_list.append(item)
        for item in delete_list:
            self.content_requested_state.pop(item)

    def _r_func(self, time):
        hit_size = 0
        for content in self.cached_content:  # 通过计算字节命中率得到reward
            for i in range(math.ceil(self.cached_content[content].should_cached_size / block_size)):
                hit_size += (sum(content_request_state[content][i][time:time + T])
                             - self.content_requested_state[content][i][time]) \
                            * block_size
        total_size = sum(request_sum_size[time:time + T]) - self.requested_sum_size[time]
        return hit_size / total_size

    def _get_evict_action(self):
        min_val = self._get_min_val()
        if request_list[self.cur_point].content_id not in self.cached_content:
            req_size = request_list[self.cur_point].should_cached_size
            req_value = 0
            for i in range(math.ceil(request_list[self.cur_point].should_cached_size / block_size)):
                req_value = X * sum(self.content_requested_state[request_list[self.cur_point].content_id][i][
                                    request_list[self.cur_point].time - T + 1:request_list[self.cur_point].time + 1])
            req_value = req_value / req_size
        else:
            req_size = request_list[self.cur_point].should_cached_size - self.cached_content[
                request_list[self.cur_point].content_id].should_cached_size
            req_value = 0
            for i in range(math.ceil(
                    self.cached_content[request_list[self.cur_point].content_id].should_cached_size / block_size),
                    math.ceil(request_list[self.cur_point].should_cached_size / block_size)):
                req_value = X * sum(self.content_requested_state[request_list[self.cur_point].content_id][i][
                                    request_list[self.cur_point].time - T + 1:request_list[self.cur_point].time + 1])
            req_value = req_value / req_size

        contents = [key for key in min_val.keys()]
        evict_candidators = {}
        candidators = {}
        candidators_number = 0
        for i in range(1,
                       10):  # 优化方案：https://leetcode-cn.com/problems/combination-sum-ii/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-3/
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
                       [[0] + candidators[k] for k, v in evict_candidators]  # state, action
            for content in contents:
                if min_val[content] == 0:
                    self._pop_item(content)
            return self._get_evict_action()
        return np.pad(np.array([0] + [req_value - v for k, v in evict_candidators]), (0, 4 - len(evict_candidators))), \
               [candidators[k] for k, v in evict_candidators]  # state, action

    def reset(self):
        print(self.name, " reset, cahce num:", len(self.cached_content), " re_size:", self.re_size, "cached size",
              sum(list(self.cached_content.values())), " cur ltime:", self.cur_time, " cur point： ", self.cur_point)
        self.done = False
        return self.state

    def render(self, mode=''):
        return None

    def close(self):
        return None

    def combinaSum(self):
        pass
