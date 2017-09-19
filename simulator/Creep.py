from .Sprite import Sprite
from .Config import Config


#Get data from https://dota2.gamepedia.com/Lane_creeps
#TODO: use json
CreepData = {}

CreepData["MeleeCreep"] = \
{
    "HP":550,
    "MP":0,
    "Speed":325,
    "Armor":2,
    "ATK":21,
    "ATKRange":100,
    "SightRange":750,
    "Bounty":36,
    "EXP":40,
    "BAT":1,
    "AS":100
}

CreepData["Radiant"] = {
    "init_loc":(-4899,-4397),
    "dest":(4165, 3681)
}

CreepData["Dire"] = {
    "init_loc":(4165, 3681),
    "dest":(-4899,-4397)
}

CreepData["visualize"] = {
    "radius":2
}

class Creep(Sprite):
    def __init__(self, Engine, side, type_name):
        super().__init__(Engine, side = side, loc = CreepData[side]["init_loc"],
                        **CreepData[type_name])
        self.dest = CreepData[self.side]["dest"]
        self.vis_r = CreepData["visualize"]["radius"]
        self.color = Config.Colors[self.side]

        if self.Engine.canvas != None:
            p = self.pos_in_wnd()
            self.v_handle = self.Engine.canvas.create_rectangle(p[0] - self.vis_r,
                            p[1] + self.vis_r,
                            p[0] + self.vis_r,
                            p[1] - self.vis_r,
                            fill = self.color)
    
    def step(self):
        if self.isAttacking():
            #in Attack animation
            return
        nearby_enemy = self.Engine.get_nearest_enemy(self)
        if len(nearby_enemy) > 0:
            #Enemy in Sight, go to attack it
            target = nearby_enemy[0]
            if target[1] < self.AttackRange:
                #target in range
                self.attack(target[0])
            else:
                #move to it
                self.set_move(target[0].location)
        else:
            #move to dest
            self.set_move(self.dest)
    
    def draw(self):
        if self.Engine.canvas != None:
            p = self.pos_in_wnd()
            self.Engine.canvas.coords(
                            self.v_handle,
                            p[0] - self.vis_r,
                            p[1] + self.vis_r,
                            p[0] + self.vis_r,
                            p[1] - self.vis_r)