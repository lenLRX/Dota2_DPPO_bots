import numpy as np
from Config import Config
from Event import Event
import math
import sys

class Sprite(object):
    def __init__(self, Engine, loc, HP, MP, Speed, Armor,
                 ATK, ATKRange, SightRange, Bounty, EXP, BAT, AS):
        self.Engine = Engine
        self.location = loc
        self.HP = HP
        self.MP = MP
        self.MovementSpeed = Speed
        self.BaseAttackTime = BAT
        self.AttackSpeed = AS
        self.Armor = Armor
        self.Attack = ATK
        self.AttackRange = ATKRange
        self.SightRange = SightRange
        self.Bounty = Bounty
        self.EXP = EXP
        self.LastAttackTime = -1
        self.AttackTime = None

        self.move_target  = None

        self._update_para()
    
    def _update_para(self):
        AttackPerSecond = self.AttackSpeed * 0.01 / self.BaseAttackTime
        self.AttackTime = 1 / AttackPerSecond
    
    def step(self):
        raise NotImplementedError
    
    def draw(self, canvas):
        raise NotImplementedError
    
    def pos_in_wnd(self):
        return (self.location[0] * Config.game2window_scale,self.location[1] * Config.game2window_scale)
    
    def attack(self,target):
        self.LastAttackTime = self.Engine.get_time()
    
    def set_move(self,target):
        self.move_target = target
    
    def move(self):
        if self.move_target is None:
            return

        dx = self.move_target[0] - self.location[0]
        dy = self.move_target[0] - self.location[1]

        a  = math.atan2(dy,dx)

        if math.isnan(a):
            print("found nan")
            sys.exit(-1)

        if not (self.move_target[0] == 0.0 and self.move_target[1] == 0.0):
            d = self.MovementSpeed * self.Engine.delta_tick
            if math.hypot(dx,dy) < d:
                self.location = self.move_target
            else:
                self.location = (self.location[0] + d *math.cos(a),
                            self.location[1] + d * math.sin(a))
            
        if self.location[0] > Config.bound_length:
            self.location[0] = Config.bound_length
        if self.location[1] > Config.bound_length:
            self.location[1] = Config.bound_length
        if self.location[0] < -Config.bound_length:
            self.location[0] = -Config.bound_length
        if self.location[1] < -Config.bound_length:
            self.location[1] = -Config.bound_length