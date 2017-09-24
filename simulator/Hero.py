from .Config import Config
from .Sprite import Sprite
from .Creep import Creep

import math

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
    "bountyEXP":200,
    "BAT":1.7,
    "AS":120
}

HeroData["Radiant"] = {
    "init_loc":(-7205, -6610)
}

HeroData["Dire"] = {
    "init_loc":(7000, 6475)
}

HeroData["visualize"] = {
    "radius":5
}

class Hero(Sprite):
    def __init__(self, Engine, side, type_name):
        super().__init__(Engine, side = side, loc = HeroData[side]["init_loc"],
                        **HeroData[type_name])
        
        self.vis_r = HeroData["visualize"]["radius"]
        self.color = Config.Colors[self.side]
        self.move_order = (0.0,0.0)

        self.last_exp = 0
        self.last_HP = self.HP

        if self.Engine.canvas != None:
            p = self.pos_in_wnd()
            self.v_handle = self.Engine.canvas.create_oval(p[0] - self.vis_r,
                            p[1] + self.vis_r,
                            p[0] + self.vis_r,
                            p[1] - self.vis_r,
                            fill = self.color)
    
    def step(self):
        p = (self.move_order[0] + self.location[0],
             self.move_order[1] + self.location[1])
        self.set_move(p)
    
    def predefined_step(self):
        nearby_ally = self.Engine.get_nearby_ally(self)
        ret = None
        if len(nearby_ally) > 0 and\
            isinstance(nearby_ally[0][0], Creep):
            ret = nearby_ally[0][0].location
            if self.side == "Radiant":
                ret = (ret[0] - 200, ret[1] - 200)
            else:
                ret = (ret[0] + 200, ret[1] + 200)
        else:
            ret = (0,0)
        dx = ret[0] - self.location[0]
        dy = ret[1] - self.location[1]

        a  = math.atan2(dy,dx)
        return (math.cos(a), math.sin(a))
    
    def get_state_tup(self):
        state = {}
        state["self_input"] = \
            [p / Config.map_div for p in self.location]
        
        nearby_ally = self.Engine.get_nearby_ally(self)
        ally_locations = [_ut[0] for _ut in nearby_ally]
        state["ally_input"] = [
            [p.location[0] / Config.map_div,p.location[1] / Config.map_div] for p in ally_locations
        ]

        if len(state["ally_input"]) == 0:
            state["ally_input"] = [[0.0, 0.0]]


        reward = \
            (self.exp - self.last_exp)\
            + (self.HP - self.last_HP)
        done = self.isDead

        self.last_exp = self.exp

        return (state, reward, done)
    
    def draw(self):
        if self.Engine.canvas != None:
            p = self.pos_in_wnd()
            self.Engine.canvas.coords(
                            self.v_handle,
                            p[0] - self.vis_r,
                            p[1] + self.vis_r,
                            p[0] + self.vis_r,
                            p[1] - self.vis_r)