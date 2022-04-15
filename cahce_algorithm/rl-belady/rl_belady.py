import collections
import numpy as np
import csv


class Node:
    def __init__(self, content_id, content_size, pre=None, next=None, nr=0, lr=0):
        self.pre = pre
        self.next = next
        self.nr = nr
        self.lr = lr
        self.freq = max(nr, lr)
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
    return head, tail


class RLBeladyCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.maxNR = 0
        self.freqMap = collections.defaultdict(create_linked_list)
        self.cache = {}
        # self.re_size_sum = 0
        self.outfile = open("D:\\Desktop\\a3c\\cahce\\cahce_algorithm\\rl_belady\\" + str(self.capacity) + ".txt",
                            "w+", newline="", encoding='UTF-8')
        self.writer = csv.writer(self.outfile, delimiter=' ')

    def delete(self, node):
        if node.pre:
            node.pre.next = node.next
            node.next.pre = node.pre
            if node.pre is self.freqMap[node.freq][0] and node.next is self.freqMap[node.freq][-1]:
                self.freqMap.pop(node.freq)
                self.maxNR = max(self.freqMap.keys())
        return node.content_id

    def update_freq(self, node, freq, nr, lr):
        self.delete(node)
        self.freqMap[freq][-1].pre.insert(node)
        if freq > self.maxNR:
            self.maxNR = freq

        head, tail = self.freqMap[self.maxNR]
        if head.next is tail:
            self.freqMap.pop(self.maxNR)
            self.maxNR = max(self.freqMap.keys())
        node.freq = freq
        node.nr = nr
        node.lr = lr

    def get(self, content_id, nr, lr):
        if content_id in self.cache:
            self.update_freq(self.cache[content_id], max(nr, lr), nr, lr)
            return 1
        return -1

    def put(self, content_id, content_size, nr, lr):
        if self.capacity != 0:
            if content_id in self.cache:
                node = self.cache[content_id]
            else:
                node = Node(content_id, content_size, nr=nr, lr=lr)
                self.cache[content_id] = node
                self.size += content_size
            while self.size > self.capacity:
                deleted = self.delete(self.freqMap[self.maxNR][0].next)
                self.size -= self.cache[deleted].content_size
                self.cache.pop(deleted)
            self.update_freq(node, max(nr, lr), nr, lr)

    def run(self, content_id, content_size, time, nr, lr, is_adm):
        # self.re_size_sum += (self.capacity-self.size) / self.capacity
        is_hit = 0
        if self.get(content_id, nr, lr) == 1:
            is_hit = 1
        else:
            if is_adm == 1:
                self.put(content_id, content_size, nr, lr)
        outinfo = [time, content_id, content_size, is_hit, is_adm]
        self.writer.writerow(outinfo)
        return is_hit

    def update_nr(self):
        for item in self.cache:
            self.update_freq(self.cache[item], max(self.cache[item].nr - 1, self.cache[item].lr + 1),
                             self.cache[item].nr - 1, self.cache[item].lr + 1)
