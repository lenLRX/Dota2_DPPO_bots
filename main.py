import argparse
import os
import sys
import time
import threading
import _thread

#import ptvsd
#ptvsd.enable_attach("dota2_bot", address = ('0.0.0.0', 3421))


from http.server import BaseHTTPRequestHandler,HTTPServer
import json

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model import Model, Shared_grad_buffers, Shared_obs_stats
from train import trainer
from test import test
from chief import chief
from utils import TrafficLight, Counter,AtomicInteger
from simulator.simulator import *
try:
    from simulator.visualizer import visualize
except Exception:
    print("running without tkinter")
finally:
    pass

dispatch_table = {}


class RequestHandler(BaseHTTPRequestHandler):

    def __init__(self,req,client,server):
        BaseHTTPRequestHandler.__init__(self,req,client,server)
    
    def log_message(self, format, *args):
        #silent
        return
            
    def do_GET(self):
        
        request_path = self.path
        
        print("\ndo_Get it should not happen\n")
        
        self.send_response(200)
        
    def do_POST(self):

        _debug = False
        
        request_path = self.path
        
        request_headers = self.headers
        content_length = request_headers.get_all('content-length')
        length = int(content_length[0]) if content_length else 0
        content = self.rfile.read(length)

        if _debug:
            print("\n----- Request Start ----->\n")
            print(request_path)
            print(request_headers)
            print(content)
            print("<----- Request End -----\n")
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(self.dispatch(content.decode("ascii")).encode("ascii"))

    def get_target(self,msg):
        obj = json.loads(msg)
        return obj["state"]["side"] , obj

    def dispatch(self,msg):
        #print(msg)
        target , json_obj = self.get_target(msg)
        agent = dispatch_table[target]
        st = json_obj
        st["state"]["self_input"] = st["state"]["self_input"][-2:]
        raw_act = agent.step((st["state"],float(st["reward"]),st["done"] == "true"))
        return "%f %f"%(raw_act[0] * 1000,raw_act[1] * 1000)

    do_PUT = do_POST
    do_DELETE = do_GET

def start_env():
    port = 8080
    print('Listening on localhost:%s' % port)
    server = HTTPServer(('', port), RequestHandler)
    server.serve_forever()

def start_simulator(num_iter):
    time.sleep(0.5)

    count = num_iter

    while True:
        count += 1
        print("%d simulated game starts!"%count)
        dire_act = [0.0,0.0]
        dire_sim = DotaSimulator(Config.dire_init_pos)
        dire_agent = dispatch_table["Dire"]

        rad_act = [0.0,0.0]
        rad_sim = DotaSimulator(Config.rad_init_pos)
        rad_agent = dispatch_table["Radiant"]

        for i in range(1000):
            d_tup = dire_sim.step(dire_act)
            #print(d_tup)
            dire_act = dire_agent.step(d_tup)

            r_tup = rad_sim.step(rad_act)
            #print(r_tup)
            rad_act = rad_agent.step(r_tup)
        
        rad_agent.waitTraningFinish()
        dire_agent.waitTraningFinish()

def start_simulator2():
    time.sleep(0.5)

    yield

    num_iter, canvas = yield

    count = num_iter

    from simulator.Hero import Hero

    def dist2mid(pos):
        return math.hypot(pos[0],pos[1])

    def dotproduct(pd_act, act, a):
        dp = float(pd_act[0]*act[0] + pd_act[1]*act[1]) * 0.1 * a
        if dp != 0:
            return dp / math.hypot(act[0],act[1])# normalize
        else:
            return dp
            

    def reward(last, now, a):
        _d = dist2mid(now)
        _ld = dist2mid(last)
        return ((_ld - _d) * 0.001 - 0.00001 * _d) * a

    discount_factor = 1.0

    while True:
        eng = DotaSimulator(Config.dire_init_pos,canvas = canvas)
        _engine = eng
        count += 1
        print("%d simulated game starts!"%count)
        dire_act = np.asarray([0.0,0.0])
        rad_act = np.asarray([0.0,0.0])
        dire_agent = dispatch_table["Dire"]
        rad_agent = dispatch_table["Radiant"]

        dire_hero = Hero(_engine, "Dire", "ShadowFiend")
        rad_hero = Hero(_engine, "Radiant", "ShadowFiend")

        last_dire_location = dire_hero.location
        last_rad_location = rad_hero.location

        _engine.add_hero(dire_hero)
        _engine.add_hero(rad_hero)

        #discount_factor -= 0.001

        if discount_factor < 0.0:
            discount_factor = 0.0

        while _engine.get_time() < 100:
            dire_hero.move_order = (dire_act[0] * 1000,dire_act[1] * 1000)
            rad_hero.move_order = (rad_act[0] * 1000,rad_act[1] * 1000)

            dire_pd_act = dire_hero.predefined_step()
            rad_pd_act = rad_hero.predefined_step()

            _engine.loop()
            if _engine.canvas != None:
                _engine.draw()
                canvas.update_idletasks()
            _engine.tick_tick()

            d_tup = dire_hero.get_state_tup()
            r_tup = rad_hero.get_state_tup()

            d_tup = (d_tup[0],
                d_tup[1] + reward(last_dire_location,
                dire_hero.location,discount_factor),
                d_tup[2])
            
            r_tup = (r_tup[0],
                r_tup[1] + reward(last_rad_location,
                rad_hero.location, discount_factor),
                r_tup[2])

            dire_act = dire_agent.step(d_tup)
            rad_act = rad_agent.step(r_tup)

            print("game %d t=%f,r_act=%s,r_reward=%f,d_act=%s,d_reward=%f"\
                %(count, _engine.get_time(),str(rad_act),r_tup[1],str(dire_act),d_tup[1]))
            
            last_dire_location = dire_hero.location
            last_rad_location = rad_hero.location

            yield

            if dire_hero.isDead or rad_hero.isDead:
                break
        
        rad_agent.waitTraningFinish()
        dire_agent.waitTraningFinish()


class Params():
    def __init__(self):
        self.batch_size = 200
        self.lr = 3e-4
        self.gamma = 0.5
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.1
        self.num_epoch = 10
        self.num_steps = 20000
        self.exploration_size = 50#make it small
        self.num_processes = 4
        self.update_treshold = 2 - 1
        self.max_episode_length = 100
        self.seed = int(time.time())
        self.num_inputs = {"self_input":2,"ally_input":2}
        self.num_outputs = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action",help = "start_server or simulator")
    parser.add_argument("--model",help = "path to pretrained model")
    parser.add_argument("-i",help = "start iteration")
    args = parser.parse_args()

    params = Params()
    torch.manual_seed(params.seed)

    traffic_light = TrafficLight()
    counter = Counter()

    shared_model = Model(params.num_inputs, params.num_outputs)

    num_iter = 0

    if args.model != None:
        shared_model.load_state_dict(torch.load(args.model))
    
    if args.i != None:
        num_iter = int(args.i)

    shared_model.share_memory()
    shared_grad_buffers = Shared_grad_buffers(shared_model)
    #shared_grad_buffers.share_memory()
    shared_obs_stats = Shared_obs_stats(params.num_inputs)
    #shared_obs_stats.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=params.lr)
    test_n = torch.Tensor([0])
    test_n.share_memory_()

    atomic_counter = AtomicInteger()
    CommonConV = threading.Barrier(3)

    rad_trainer = trainer(params,
    shared_model,shared_grad_buffers,shared_obs_stats,
    atomic_counter,CommonConV)

    dire_trainer = trainer(params,
    shared_model,shared_grad_buffers,shared_obs_stats,
    atomic_counter,CommonConV)

    dispatch_table["Radiant"] = rad_trainer
    dispatch_table["Dire"] = dire_trainer

    _thread.start_new_thread(chief,
    (params,CommonConV,atomic_counter,
    shared_model,shared_grad_buffers,optimizer,num_iter))

    _thread.start_new_thread(rad_trainer.loop,())
    _thread.start_new_thread(dire_trainer.loop,())

    if args.action == "start_server":
        start_env()
    elif args.action == "simulator":
        start_simulator(num_iter / params.num_epoch)
    elif args.action == "simulator2_viz":
        g = start_simulator2()
        g.send(None)
        visualize(g, num_iter / params.num_epoch)
        #print("argument error")
    elif args.action == "simulator2":
        g = start_simulator2()
        g.send(None)
        g.send(None)
        g.send((num_iter / params.num_epoch, None))
        while True:
            g.send(None)
