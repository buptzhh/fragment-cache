import gym
from gym import spaces
import numpy as np
import copy
import time
from rl_belady import RLBeladyCache
from fnn import NeuralNetwork

T = 60
test_time = 3600
max_time = 3600
K = 10000
L = 2000
RLBelady = None


class Simulator(gym.Env):
    # action_bound = [0, 1, 2, 3, 4]  # 0为不准入，1-5为准入且剔除价值量最少的4种剔除方式
    def init(self, name, maxsize):
        self.name = name
        self.done = False
        self.cache_init_isFinish = False
        self.infilepath = "G:\\dance\\caching\\rl-belady\\data\\175_new.csv"  # '../dataset/122.228.93.41.csv'
        self.future_file_path = "G:\\dance\\caching\\rl-belady\\data\\175_future_and_perdition.csv"
        self.infile = open(self.infilepath)
        self.fufile = open(self.future_file_path)
        self.cache_size = maxsize
        self.re_size = maxsize
        self.cur_point = 0
        self.cur_time = 0
        self.line_count = 0
        # value
        self.cached_content = {}
        self.delay_req = {}

        self.cur_content = None
        self.cur_time_pos = 0
        self.re_size_sum = 0

        self.model_list = []
        nn = NeuralNetwork()
        # Add Layers (Input layer is created by default)
        nn.add_layer((5, 9))
        nn.add_layer((9, 1))
        self.model_list.append(nn)

        self.s_window = []
        self.s_real_nr = []
        self.s_time = []
        self.s_label = []
        self.step = 0
        self.e_time = 0
        self.hit_times = [0, 0]
        self.is_train = True
        self.total_hits = 0
        pass

    def read_a_line(self):
        line = self.infile.readline()
        info = line.split(' ')
        ltime = int(float(info[0]))
        content_id = int(info[1])
        content_size = int(info[2])
        lr = int(info[3])
        delat = [int(float(i)) for i in info[4].split(',')[:4]]

        if ltime >= max_time:
            print('read finish')
            self.infile.close()
            return 0, 0, 100000000, 0, 0, 0, 0
        line = self.fufile.readline()
        info = line.split(' ')
        real_future_nr = int(info[3])
        if real_future_nr < 0 :
            real_future_nr = 3600
        nr = int(float(info[4]))
        return content_id, content_size, ltime, nr, delat, lr, real_future_nr

    def next_request(self):
        self.cur_point += 1
        if self.cur_point % 100000 == 0:
            print(self.cur_point)
        req_id, _content_size, req_time, nr, delat, lr, real_future_nr = self.read_a_line()
        if req_time >= max_time:
            self.done = True
            return True
        # self.re_size_sum += self.re_size / MAXSIZE

        if req_time != self.cur_time:
            RLBelady.update_nr()

        input = delat + [nr]
        is_adm = 0
        if self.step == 0:
            is_adm = 1
        else:
            is_adm = self.model_list[-1].predict(np.asarray(input).reshape(1, 5))
            if is_adm > 0.5:
                is_adm = 1
            else:
                is_adm = 0
        is_hit =  RLBelady.run(req_id, _content_size, req_time, nr, lr, is_adm)
        self.hit_times[1] += is_hit
        self.total_hits += is_hit
        if len(self.s_window) < K + L:
            self.s_window.append(input)
            self.s_label.append(is_adm)
            self.s_real_nr.append(real_future_nr)
            self.s_time.append(req_time)
            if len(self.s_window) > K and self.is_train:
                self.label_update()
                self.model_list.append(copy.deepcopy(self.model_list[-1]))
                train_time = time.time()
                self.model_list[-1].train(np.asarray(self.s_window).reshape(len(self.s_label), 5, 1),
                                          np.asarray(self.s_label), 10)
                end_time = time.time()
                print(end_time - train_time)
                self.is_train = False
        else:
            self.s_label.clear()
            self.s_real_nr.clear()
            self.s_window.clear()
            self.s_time.clear()
            if self.hit_times[1] >= self.hit_times[0]:
                self.e_time +=1
                if len(self.model_list) > 5:
                    self.model_list.pop(0)
            else:
                self.model_list.pop(-1)
            self.hit_times[0] = self.hit_times[1]
            self.hit_times[1] = 0
            self.step += 1
            self.is_train = True
        return False

    def label_update(self):
        # TODO：LSO更新
        L = []
        for i in range(len(self.s_window)):
            l = 0
            for j in range(self.s_real_nr[i]):
                if self.s_time[j] - self.s_time[i] + self.s_real_nr[j] <= self.s_real_nr[i]:
                    l += 1
            L.append(l)
        L_sorted = sorted(L)
        L1 = L_sorted[500]
        L2 = L_sorted[9500]
        for i in range(len(self.s_label)):
            if L[i] <= L1:
                self.s_label[i] = 1
            if L[i] >= L2:
                self.s_label[i] = 0
        pass


if __name__ == "__main__":
    SIZE = [204800000, 512000000, 1024000000, 2048000000, 4096000000, 8192000000]  # {2048000000,20480000000}

    for MAXSIZE in SIZE:
        train_time = time.time()
        RLBelady = RLBeladyCache(MAXSIZE)
        simulator = Simulator()
        simulator.init("worker", MAXSIZE)

        done = False
        while not done:
            done = simulator.next_request()
        print("sim_etime"+str(simulator.e_time) + "hit:"+str(simulator.total_hits))
        end_time = time.time()
        print("run_time"+str(end_time - train_time))