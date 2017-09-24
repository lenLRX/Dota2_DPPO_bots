from .Sprite import Sprite
from .Config import Config

TowerData = {}

TowerData["Tier1Tower"] = {}

TowerData["Tier1Tower"]["data"] = {
    "HP":1400,
    "MP":0,
    "Speed":0,
    "Armor":14,
    "ATK":120,#110~120
    "ATKRange":700,
    "SightRange":1900,
    "Bounty":36,
    "bountyEXP":0,
    "BAT":1,
    "AS":100
}

TowerData["Tier1Tower"]["Radiant"] = {
    "init_loc":(-1661,-1505),
    "dest":(None, None)
}

TowerData["Tier1Tower"]["Dire"] = {
    "init_loc":(1032, 359),
    "dest":(None, None)
}

TowerData["Tier1Tower"]["visualize"] = {
    "radius":5
}

class Tower(Sprite):
    def __init__(self, Engine, side, type_name, loc):
        super().__init__(Engine, side = side, loc = loc,
                        **TowerData[type_name]["data"])
        self.type_name = type_name
        self.vis_r = TowerData[self.type_name]["visualize"]["radius"]
        self.color = Config.Colors[self.side]

        if self.Engine.canvas != None:
            p = self.pos_in_wnd()
            self.v_handle = self.Engine.canvas.create_rectangle(p[0] - self.vis_r,
                            p[1] + self.vis_r,
                            p[0] + self.vis_r,
                            p[1] - self.vis_r,
                            fill = self.color)
    #tower will not move, just attack nearby enemy
    def step(self):
        if self.isAttacking():
            #in Attack animation
            return
        nearby_enemy = self.Engine.get_nearby_enemy(self)
        if len(nearby_enemy) > 0:
            #Enemy in Sight, go to attack it
            target = nearby_enemy[0]
            if target[1] < self.AttackRange:
                #target in range
                self.attack(target[0])
    #no need to redraw tower
    def draw(self):
        pass