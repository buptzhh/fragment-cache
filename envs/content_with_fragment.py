# coding=UTF-8
from numpy import random
import numpy as np
import scipy.stats as stats
import math

# init_fragment_size = dict({1: 102400, 2: 79872, 3: 62464, 4: 50176, 5: 39936, 6: 31744, 7: 26624, 8: 21504, 9: 17408,
#                            10: 14336, 11: 11264, 12: 10240, 13: 8192, 14: 7168, 15: 6144, 16: 5120, 17: 4096,
#                            18: 3072, 19: 3072, 20: 3072, 21: 2048, 22: 2048, 23: 2048,
#                            24: 1024, 25: 1024, 26: 1024, 27: 1024, 28: 1024, 29: 1024, 30: 1024, 31: 1024, 32: 1024,
#                            33: 1024})
# init_fragment_size = {1: 102400, 2: 79872, 3: 63488, 4: 51200, 5: 40960, 6: 32768, 7: 26624, 8: 20480, 9: 16384, 10: 14336, 11: 12288, 12: 10240, 13: 8192, 14: 6144, 15: 6144, 16: 4096, 17: 4096, 18: 4096, 19: 4096, 20: 2048, 21: 2048, 22: 2048, 23: 2048, 24: 2048, 25: 2048, 26: 2048, 27: 2048}
block_size = 10240
fragment_min_size = 51200
fragment_max_size = 1024000
alpha = 1.01
zipf = [(1 / (i ** alpha)) for i in range(1, 2501)]  # 为请求到每一块的概率
print(sum(zipf))
# fragment size in [10240, 1024000]
def get_block_num(size):
    return math.ceil(size / block_size)


class content_with_fragment():
    def __init__(self, content_id, size, time):
        self.content_id = content_id
        self.size = size
        self.time = time

        self.block_num = get_block_num(size=self.size)
        # self.block_value = np.zeros((1, self.block_num))[0]
        request_por = random.zipf(a=1.1, size=1)
        if request_por > 100:
            request_por = 100
        self.request_size = math.ceil(request_por / 100 * self.size)
        if self.request_size <= 5 * block_size:
            self.request_size = 5 * block_size
        self.should_cached_size = math.ceil(self.request_size / block_size) * block_size
        self.last_fragment_size = 0
        self.zipf_value = sum(zipf[:self.block_num])

    def init(self):
        pass
        # print(np.random.zipf(a=1.05, size=100))
        # print(request_por)
        # 初始分段
        # init_fragment_size = self.init_fragment_size1(self.size)
        # for i in list(reversed(range(1, len(init_fragment_size) + 1))):
        #     if self.should_cached_size <= self.request_size:
        #         self.should_cached_size += init_fragment_size[i]
        #         if i == 1:
        #             self.last_fragment_size = init_fragment_size[i]
        #     else:
        #         self.last_fragment_size = init_fragment_size[i + 1]
        #         break


#     def init_fragment_size1(self, _whole_size):
#         a = 1.0001
#         block = []
#         block_num = get_block_num(_whole_size)
#         for i in range(1, block_num+1):
#             if _whole_size >= block_size:
#                 block.append([block_size, 1 / (i ** a)])  # * sum([j ** (-a) for j in range(1, i + 1)])))
#                 _whole_size -= block_size
#             else:
#                 block.append([_whole_size, 1 / (i ** a)])
#         fragment = {}
#         fragment_num = 1
#         block_value = sum([block[i][1] for i in range(block_num - math.ceil(fragment_max_size/block_size), block_num)])
#         print(block_value)
#         local_size = 0
#         local_value = 0
#         print(len(block))
#         for i in list(reversed(block)):
#             local_size += i[0]
#             local_value += i[1]
#             if local_value >= block_value:
#                 if local_size < fragment_min_size:
#                     continue
#                 fragment[fragment_num] = local_size
#                 local_size = 0
#                 local_value = 0
#                 fragment_num += 1
#         fragment[1] += local_size
#         print(fragment)
#         return fragment
#
# print(random.zipf(a=1.1, size=100))