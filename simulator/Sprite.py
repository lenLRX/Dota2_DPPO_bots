import numpy as np
from .Config import Config
from .Event import Event
from .Events import *

import math
import sys

class Sprite(object):
    def __init__(self, Engine, side, loc, HP, MP, Speed, Armor,
                 ATK, ATKRange, SightRange, Bounty, bountyEXP, BAT, AS):
        self.Engine = Engine
        self.canvas = self.Engine.canvas
        self.v_handle = None
        self.side = side
        self.isDead = False
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
        self.bountyEXP = bountyEXP
        self.LastAttackTime = -1
        self.AttackTime = None
        self.exp = 0

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
        return (self.location[0] * Config.game2window_scale * 0.5 + Config.windows_size * 0.5,
            self.location[1] * Config.game2window_scale * 0.5 + Config.windows_size * 0.5)
    
    def attack(self,target):
        self.LastAttackTime = self.Engine.get_time()
        AttackEvent.Create(self,target)
    
    def isAttacking(self):
        return self.Engine.get_time() - self.LastAttackTime < self.AttackTime
    
    def set_move(self,target):
        self.move_target = target
    
    def move(self):
        if self.move_target is None:
            return

        #Sprite cant move when it is attacking
        if self.isAttacking():
            return

        dx = self.move_target[0] - self.location[0]
        dy = self.move_target[1] - self.location[1]

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
            self.location = (Config.bound_length, self.location[1])
        if self.location[1] > Config.bound_length:
            self.location = (self.location[0], Config.bound_length)
        if self.location[0] < -Config.bound_length:
            self.location = (-Config.bound_length, self.location[1])
        if self.location[1] < -Config.bound_length:
            self.location = (self.location[0], -Config.bound_length)
    
    def damadged(self, dmg, dmg_type = None):
        #TODO
        if self.isDead:
            return False
        self.HP -= dmg
        if self.HP <= 0.0:
            print("I'm Dead")
            self.dead()
        return True
    
    def dead(self):
        self.isDead = True
        if self.v_handle != None:
            self.canvas.delete(self.v_handle)
        for s in self.Engine.sprites:
            if Sprite.S2Sdistance(self, s) <= 1300:
                s.exp += self.bountyEXP

    
    @staticmethod
    def S2Sdistance(s1,s2):
        dx = s1.location[0] - s2.location[0]
        dy = s1.location[1] - s2.location[1]
        return math.sqrt(dx * dx + dy * dy)
    