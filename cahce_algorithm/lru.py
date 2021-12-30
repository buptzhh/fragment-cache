import numpy as np

class DLinkedNode:
    def __init__(self, content_id=0, content_size=0):
        self.content_id = content_id
        self.content_size = content_size
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int, test_time:int):
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0
        self.hit_time = np.zeros((1, test_time))[0]
        self.total_time = np.zeros((1, test_time))[0]
        self.hit_size = np.zeros((1, test_time))[0]
        self.total_size = np.zeros((1, test_time))[0]
        self.back_size = np.zeros((1, test_time))[0]
        # self.re_size_sum = 0

    def get(self, content_id: int) -> int:
        if content_id not in self.cache:
            return -1
        # 如果 content_id 存在，先通过哈希表定位，再移到头部
        node = self.cache[content_id]
        self.moveToHead(node)
        return 1

    def put(self, content_id: int, content_size: int) -> None:
        if content_id not in self.cache:
            # 如果 content_id 不存在，创建一个新的节点
            node = DLinkedNode(content_id, content_size)
            # 添加进哈希表
            self.cache[content_id] = node

            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += content_size

            while self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.content_id)
                self.size -= removed.content_size
        else:
            # 如果 content_id 存在，先通过哈希表定位，再修改 content_size，并移到头部
            node = self.cache[content_id]
            node.content_size = content_size
            self.moveToHead(node)

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

    def run(self, content_id: int, content_size: int, time: int, req_size: int):
        self.total_time[time] += 1
        self.total_size[time] += req_size
        # self.re_size_sum += (self.capacity-self.size) / self.capacity
        if self.get(content_id) == 1:
            self.hit_time[time] += 1
            self.hit_size[time] += req_size
        else:
            self.back_size[time] += content_size
            self.put(content_id, content_size)

    def get_result(self):
        return self.total_time, self.hit_time, self.total_size, self.hit_size, self.back_size  # , self.re_size_sum
