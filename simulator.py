#we can use simulator to train the bots
import math
import sys
class Config(object):
    def __init__(self):
        self.delta_time = 1.0
        self.map_div = 7000.0
        self.rad_init_pos = [-0.95714285714286 * self.map_div,
            -0.95714341517857 * self.map_div]
        self.dire_init_pos = [0.98571428571429 * self.map_div,
            0.949999441964297 * self.map_div]
        self.velocity = 315


class DotaSimulator(object):
    def __init__(self,init_pos):
        self.config = Config()
        self.self_input = [38,1.2000000476837,1,500,500,273,273,45,0,0,0,None,None]
        self.pos = init_pos[:]
        self.self_input[-1] = self.pos[-1] / self.config.map_div
        self.self_input[-2] = self.pos[-2] / self.config.map_div
        self.last_d = self.d()
    
    def d(self):
        dist2_0_0 = math.hypot(self.pos[0],self.pos[1])
        dist2midline = abs( (self.pos[0] - self.pos[1]) / math.sqrt(2))
        return dist2_0_0

    def reward(self):
        _d = self.d()
        r = (self.last_d - _d) / 100.0 - _d / 10000.0
        self.last_d = _d
        return r
    
    def step(self,action):
        a  = math.atan2(*action)

        if math.isnan(a):
            print("found nan")
            sys.exit(-1)
        
        if not (action[0] == 0.0 and action[1] == 0.0):
            self.pos[0] += self.config.velocity * math.cos(a) * self.config.delta_time
            self.pos[1] += self.config.velocity * math.sin(a) * self.config.delta_time

        if self.pos[0] > 7000.0:
            self.pos[0] = 7000.0
        if self.pos[1] > 7000.0:
            self.pos[1] = 7000.0
        if self.pos[0] < -7000.0:
            self.pos[0] = -7000.0
        if self.pos[1] < -7000.0:
            self.pos[1] = -7000.0

        self.self_input[-1] = self.pos[-1] / self.config.map_div
        self.self_input[-2] = self.pos[-2] / self.config.map_div

        state = {
            "self_input":self.self_input,
            "ally_input":[[0,0,0,0,0,0,0]]
        }

        reward = self.reward()

        done = False

        return (state,reward,done)
    