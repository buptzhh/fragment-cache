import collections
import numpy as np

class Node:
    def __init__(self, content_id, content_size, pre=None, next=None, freq=0):
        self.pre = pre
        self.next = next
        self.freq = freq
        self.content_size = content_size
        self.content_id = content_id

    def insert(self, next):
        next.pre = self
        next.next = self.next
        self.next.pre = next
        self.next = next


def create_linked_list():
    head = Node(0, 0)
    tail = Node(0, 0)
    head.next = tail
    tail.pre = head
    return (head, tail)


class LFUCache:
    def __init__(self, capacity, test_time):
        self.capacity = capacity
        self.size = 0
        self.minFreq = 1
        self.freqMap = collections.defaultdict(create_linked_list)
        self.cache = {}
        self.hit_time = np.zeros((1, test_time))[0]
        self.total_time = np.zeros((1, test_time))[0]
        self.hit_size = np.zeros((1, test_time))[0]
        self.total_size = np.zeros((1, test_time))[0]
        self.back_size = np.zeros((1, test_time))[0]
        # self.re_size_sum = 0

    def delete(self, node):
        if node.pre:
            node.pre.next = node.next
            node.next.pre = node.pre
            if node.pre is self.freqMap[node.freq][0] and node.next is self.freqMap[node.freq][-1]:
                self.freqMap.pop(node.freq)
                self.minFreq = min(self.freqMap.keys())
        return node.content_id

    def update_freq(self, node, req_history):
        self.delete(node)
        self.freqMap[req_history][-1].pre.insert(node)
        if req_history < self.minFreq:
            self.minFreq = req_history
        elif self.minFreq == req_history:
            head, tail = self.freqMap[self.minFreq]
            if head.next is tail:
                self.freqMap.pop(self.minFreq)
                self.minFreq = min(self.freqMap.keys())
        node.freq = req_history

    def get(self, content_id, req_history):
        if content_id in self.cache:
            self.update_freq(self.cache[content_id], req_history)
            return 1
        return -1

    def put(self, content_id, content_size, req_history):
        if self.capacity != 0:
            if content_id in self.cache:
                node = self.cache[content_id]
            else:
                node = Node(content_id, content_size, freq=req_history)
                self.cache[content_id] = node
                self.size += content_size
            while self.size > self.capacity:
                deleted = self.delete(self.freqMap[self.minFreq][0].next)
                self.size -= self.cache[deleted].content_size
                self.cache.pop(deleted)
            self.update_freq(node, req_history)

    def run(self, content_id, content_size, time, req_size, req_history):
        self.total_time[time] += 1
        self.total_size[time] += req_size
        # self.re_size_sum += (self.capacity-self.size) / self.capacity
        if self.get(content_id, req_history) == 1:
            self.hit_time[time] += 1
            self.hit_size[time] += req_size
        else:
            self.back_size[time] += content_size
            self.put(content_id, content_size, req_history)

    def get_result(self):
        return self.total_time, self.hit_time, self.total_size, self.hit_size, self.back_size  # , self.re_size_sum
