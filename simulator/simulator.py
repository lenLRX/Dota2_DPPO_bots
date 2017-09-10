#we can use simulator to train the bots
import math
import sys
from Config import Config
from Event import Event, EventQueue
from Creep import Creep

def spawn_fn(engine):
    engine.sprites += [Creep("Radiant","MeleeCreep") for i in range(5)]
    engine.sprites += [Creep("Dire","MeleeCreep") for i in range(5)]

class DotaSimulator(object):
    def __init__(self,init_pos):
        self.self_input = [None,None]
        self.pos = init_pos[:]
        self.self_input[-1] = self.pos[-1] / Config.map_div
        self.self_input[-2] = self.pos[-2] / Config.map_div
        self.last_d = self.d()

        self.delta_tick = 1.0 / Config.tick_per_second
        self.tick_time = 0.0
        self.event_queue = EventQueue()
        self.sprites = []

        self.event_queue.enqueue(Event(30.0,spawn_fn,self))
    
    def d(self):
        dist2_0_0 = math.hypot(self.pos[0],self.pos[1])
        dist2midline = abs( (self.pos[0] - self.pos[1]) / math.sqrt(2))
        return dist2_0_0

    def reward(self):
        _d = self.d()
        r = (self.last_d - _d) / 100.0 - _d / 10000.0
        self.last_d = _d
        return r

    def draw(self, canvas):
        for sprite in self.sprites:
            sprite.draw(canvas)

    def loop(self):
        #process events
        while True:
            event = self.event_queue.fetch(self.tick_time)
            if event != None:
                event.activate()
            else:
                break
        
        for sprite in self.sprites:
            sprite.step()
        
        for sprite in self.sprites:
            sprite.move()
    
    def step(self,action):
        a  = math.atan2(*action)

        if math.isnan(a):
            print("found nan")
            sys.exit(-1)
        
        if not (action[0] == 0.0 and action[1] == 0.0):
            self.pos[0] += Config.velocity * math.cos(a) * Config.delta_time
            self.pos[1] += Config.velocity * math.sin(a) * Config.delta_time

        if self.pos[0] > Config.bound_length:
            self.pos[0] = Config.bound_length
        if self.pos[1] > Config.bound_length:
            self.pos[1] = Config.bound_length
        if self.pos[0] < -Config.bound_length:
            self.pos[0] = -Config.bound_length
        if self.pos[1] < -Config.bound_length:
            self.pos[1] = -Config.bound_length

        self.self_input[-1] = self.pos[-1] / Config.map_div
        self.self_input[-2] = self.pos[-2] / Config.map_div

        state = {
            "self_input":self.self_input,
            "ally_input":[[0,0,0,0,0,0,0]]
        }

        reward = self.reward()

        done = False

        return (state,reward,done)
    