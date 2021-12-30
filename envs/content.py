from numpy import random
import numpy as np
import scipy.stats as stats

init_fragment_size = dict({1: 2048, 2: 9216, 3: 41984, 4: 187392, 5: 840704, 6: 4037632})
# init_fragment_size = dict({1: 2048, 2: 9216, 3: 41984, 4: 187392, 5: 840704, 6: 4037632})

class content():
    def __init__(self, content_id, size, time):
        self.content_id = content_id
        self.size = size
        self.time = time