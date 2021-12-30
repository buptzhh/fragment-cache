import gym
from gym import spaces
import numpy as np
from .content import content
from itertools import combinations
import copy
# only for s1dataset
MAXSIZE = 104857600
T = 40
test_time = 7200
X = 100000
EP_STEP = 500
request_list = []

content_request_state = {}  # content = {content_id:[time0_req,time1_req,...]}
content_requested_state = {}

request_sum = np.zeros((1, test_time))[0]
requested_sum = np.zeros((1, test_time))[0]

request_sum_size = np.zeros((1, test_time))[0]
requested_sum_size = np.zeros((1, test_time))[0]

cache_init_isFinish = False
cache_size = MAXSIZE
re_size = MAXSIZE
cahced_content = {}
start_point = 0
start_time = 0
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
        request_list.append(content(content_id, content_size, time))
        if content_id not in content_request_state:
            content_request_state[content_id] = np.zeros((1, test_time))[0]
            content_requested_state[content_id] = np.zeros((1, test_time))[0]
        content_request_state[content_id][time] += 1
        request_sum[time] += 1
        request_sum_size[time] += content_size
        if not cache_init_isFinish:
            content_requested_state[content_id][time] += 1
            requested_sum_size[time] += content_size
            requested_sum[time] += 1
            if content_size <= re_size:  # 初始化
                if content_id not in cahced_content:
                    cahced_content[content_id] = content_size
                    re_size -= content_size
            if time >= T:
                start_point = line_count
                start_time = time
                cache_init_isFinish = True
                print(len(cahced_content))
    print('read finish')


class CacheSimulator(gym.Env):
    # action_bound = [0, 1, 2, 3, 4]  # 0为不准入，1-5为准入且剔除价值量最少的4种剔除方式
    def __init__(self,name):
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
        self.cached_content = copy.deepcopy(cahced_content)

        self.content_requested_state = copy.deepcopy(content_requested_state)
        self.requested_sum_size = copy.deepcopy(requested_sum_size)
        self.requested_sum = copy.deepcopy(requested_sum)

        self._next_request()
        while request_list[self.cur_point].content_id in self.cached_content:
            self._next_request()
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
            self._cache_item(request_list[self.cur_point].content_id, request_list[self.cur_point].size)

        self._next_request()
        self.state, self.evitc_action = self._get_evict_action()
        while len(self.evitc_action) == 0 or self.re_size > request_list[self.cur_point].size \
                or request_list[self.cur_point].content_id in self.cached_content:  # 缓存还有空间和缓存命中的情况
            if request_list[self.cur_point].content_id not in self.cached_content and len(self.evitc_action) != 0:
                self._cache_item(request_list[self.cur_point].content_id, request_list[self.cur_point].size)
            self._next_request()
            self.state, self.evitc_action = self._get_evict_action()
        reward = self._r_func(self.cur_time)
        print(self.state)
        print(self.evitc_action)
        return self.state, reward, self.done

    def _pop_item(self, _content_id):
        self.re_size += self.cached_content[_content_id]
        self.cached_content.pop(_content_id)
        self.content_fragment_value.pop(_content_id)

    def _cache_item(self, _content_id, _content_size):
        self.cached_content[_content_id] = _content_size
        self.re_size -= _content_size
        self.content_fragment_value[_content_id] = X * sum(
            self.content_requested_state[_content_id][self.cur_time - T + 1:self.cur_time + 1]) / _content_size

    def _next_request(self):
        self.cur_point += 1
        req_id = request_list[self.cur_point].content_id
        req_size = request_list[self.cur_point].size
        req_time = request_list[self.cur_point].time
        if req_time != self.cur_time:
            self.cur_time = req_time
            self._update_value(self.cur_time)
        if req_id not in self.content_requested_state:
            self.content_requested_state[req_id] = np.zeros((1, test_time))[0]
        if req_id in self.cached_content:  # 更新已经缓存内容的价值
            self.content_fragment_value[req_id] = X * sum(
                self.content_requested_state[req_id][req_time - T + 1:req_time + 1]) / req_size
        self.content_requested_state[req_id][req_time] += 1
        self.requested_sum_size[req_time] += req_size
        self.requested_sum[self.cur_time] += 1

    def _get_min_val(self):
        sorted_content = sorted(self.content_fragment_value.items(), key=lambda x: x[1], reverse=False)
        sorted_content = sorted_content[:10]
        min_val = {}
        for content, value in sorted_content:
            min_val[content] = value
        return min_val

    def _update_value(self, time):
        self.content_fragment_value = {}
        for item in self.cached_content:
            self.content_fragment_value[item] = X * sum(
                self.content_requested_state[item][time - T + 1:time + 1]) / self.cached_content[item]
        for item in self.content_requested_state:
            if item not in self.cached_content and sum(self.content_requested_state[item][time - T + 1:time + 1]) == 0:
                self.content_requested_state.pop(item)

    def _r_func(self, time):
        hit_time = 0
        for content in self.cached_content:  # 通过计算字节命中率得到reward
            hit_time += (sum(content_request_state[content][time:time + T]) - self.content_requested_state[content][
                time]) * self.cached_content[content]
        total_time = sum(request_sum_size[time:time + T]) - self.requested_sum_size[time]
        #     hit_time += sum(content_request_state[content][ltime:ltime + T]) - self.content_requested_state[content][ltime]
        # total_time = sum(request_sum[ltime:ltime + T]) - self.requested_sum[ltime]
        # print(hit_time / total_time)
        return hit_time / total_time

    def _get_evict_action(self):
        min_val = self._get_min_val()
        req_size = request_list[self.cur_point].size
        req_value = X * sum(self.content_requested_state[request_list[self.cur_point].content_id][
                request_list[self.cur_point].time - T + 1:request_list[self.cur_point].time + 1]) / request_list[self.cur_point].size
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
                    total_size += self.cached_content[item]
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
                return np.pad(np.array([0] + [req_value-v for k, v in evict_candidators]),(0, 4 - len(evict_candidators))), \
                       [candidators[k] for k, v in evict_candidators]  # state, action
            for content in contents:
                if min_val[content] == 0:
                    self._pop_item(content)
            return self._get_evict_action()
        return np.pad(np.array([0] + [req_value-v for k, v in evict_candidators]), (0, 4 - len(evict_candidators))), \
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
