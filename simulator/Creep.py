from Sprite import Sprite
from Config import Config


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
    "init_loc":(-4899 / Config.map_div,-4397 / Config.map_div),
    "dest":(4165 / Config.map_div, 3681 / Config.map_div)
}

CreepData["Dire"] = {
    "init_loc":(4165 / Config.map_div, 3681 / Config.map_div),
    "dest":(-4899 / Config.map_div,-4397 / Config.map_div)
}

CreepData["visualize"] = {
    "radius":2
}

class Creep(Sprite):
    def __init__(self, side, type_name):
        super().__init__(loc = CreepData[side]["init_loc"],
                        **CreepData[type_name])
        self.side = side
        self.dest = CreepData[self.side]["dest"]
        self.vis_r = CreepData["visualize"]["radius"]
        self.color = Config.Colors[self.side]
    
    def step(self):
        if self.Engine.get_time() - self.LastAttackTime < self.AttackTime:
            #in Attack animation
            return
        nearby_enemy = self.Engine.get_nearest_enemy(self.SightRange)
        if len(nearby_enemy) > 0:
            #Enemy in Sight, go to attack it
            target = nearby_enemy[0]
            if self.Engine.get_unit2unit_distance(self,target) < self.AttackRange:
                #target in range
                self.attack(target)
            else:
                #move to it
                self.move(target.location)
        else:
            #move to dest
            self.move(self.dest)
    
    def draw(self, canvas):
        p = self.pos_in_wnd()
        canvas.create_rectangle(p[0] - self.vis_r,
                           p[1] + self.vis_r,
                           p[0] + self.vis_r,
                           p[1] - self.vis_r,
                           fill = self.color)