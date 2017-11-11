import torch
import torch.multiprocessing as mp
import threading
import time
import math

class AtomicInteger:
    def __init__(self):
        self.val = 0
        self.lock = threading.Lock()
    
    def getVal(self):
        return self.val

    def setVal(self,val):
        self.lock.acquire()
        self.val = val
        self.lock.release()

    def inc(self):
        self.lock.acquire()
        self.val += 1
        self.lock.release()

class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)

class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0

def dist2mid(pos):
    return math.hypot(pos[0],pos[1])

def dotproduct(pd_act, act, a):
    dp = float(pd_act[0]*act[0] + pd_act[1]*act[1]) * a
    bias = 0.2
    if dp != 0:
        return dp / math.hypot(act[0],act[1]) - bias# normalize
    else:
        return dp - bias
    
def hero_location_by_tup(t):
    return t[0][:2]

def get_action(idx):
    x,y = (idx // param.num_outputs, idx % param.num_outputs)
    return [float(x - param.num_outputs // 2), float(y - param.num_outputs // 2)]
            

def reward(last, now, a):
    _d = dist2mid(now)
    _ld = dist2mid(last)
    return ((_ld - _d) * 0.01) * a

class Params():
    def __init__(self):
        self.batch_size = 200000
        self.game_duriation = 2
        self.tick_per_action = 2
        self.game_per_update = 1
        self.lr = 1e-5
        self.gamma = 0.0
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.1
        self.num_epoch = 1
        self.num_steps = 20000
        self.exploration_size = 50#make it small
        self.update_treshold = 2 - 1
        self.max_episode_length = 100
        self.seed = int(time.time())
        self.num_inputs = {"self_input":3,"ally_input":6}
        self.num_outputs = 3
        self.log_std_bound = 1
        self.use_lstm = False

param = Params()
