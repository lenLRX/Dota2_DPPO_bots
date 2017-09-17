from .Config import Config
from .Sprite import Sprite

HeroData = {}

HeroData["ShadowFiend"] = \
{
    "HP":500,
    "MP":273,
    "Speed":315,
    "Armor":0.86,
    "ATK":21,
    "ATKRange":500,
    "SightRange":1800,
    "Bounty":200,
    "EXP":200,
    "BAT":1.7,
    "AS":120
}

HeroData["Radiant"] = {
    "init_loc":(-7205 / Config.map_div, -6610 / Config.map_div)
}

HeroData["Dire"] = {
    "init_loc":(7000 / Config.map_div, 6475 / Config.map_div)
}

HeroData["visualize"] = {
    "radius":5
}

class Hero(Sprite):
    def __init__(self, side, type_name):
        super().__init__(loc = HeroData[side]["init_loc"],
                        **HeroData[type_name])
        
        self.side = side
        self.vis_r = HeroData["visualize"]["radius"]
        self.color = Config.Colors[self.side]
        self.move_order = (0.0,0.0)
    
    def step(self):
        p = (self.move_order[0] + self.location[0],
             self.move_order[1] + self.location[1])
        self.set_move(p)
    
    def draw(self, canvas):
        p = self.pos_in_wnd()
        canvas.create_oval(p[0] - self.vis_r,
                           p[1] + self.vis_r,
                           p[0] + self.vis_r,
                           p[1] - self.vis_r,
                           fill = self.color)