import argparse
import os
import sys
import time
import threading
import _thread

from cppSimulator.cppSimulator import *

#import ptvsd
#ptvsd.enable_attach("dota2_bot", address = ('0.0.0.0', 3421))


from http.server import BaseHTTPRequestHandler,HTTPServer
import json

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from model import Model, Shared_grad_buffers
from train import trainer
from chief import chief
from utils import *
from simulator.simulator import *
from mp_trainer import mp_trainer

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
    raise NotImplementedError

def start_simulator2():
    raise NotImplementedError

def start_cppSimulator():
    time.sleep(0.5)

    yield

    num_iter, canvas = yield

    count = num_iter

    discount_factor = 100.0

    while True:
        _engine = cppSimulator(canvas)
        count += 1
        print("%d simulated game starts!"%count)
        dire_act = np.asarray([0.0,0.0])
        rad_act = np.asarray([0.0,0.0])
        dire_agent = dispatch_table["Dire"]
        rad_agent = dispatch_table["Radiant"]

        shared_model = dire_agent.shared_model

        d_tup = _engine.get_state_tup("Dire", 0)
        r_tup = _engine.get_state_tup("Radiant", 0)

        last_dire_location = hero_location_by_tup(d_tup)
        last_rad_location = hero_location_by_tup(r_tup)

        r_total_reward = 0.0
        d_total_reward = 0.0

        #discount_factor -= 0.001

        dire_agent.pre_train()
        rad_agent.pre_train()

        if discount_factor < 0.0:
            discount_factor = 0.0

        tick = 0

        p_dire_act = _engine.predefined_step("Dire",0)
        p_rad_act = _engine.predefined_step("Radiant",0)

        while _engine.get_time() < param.game_duriation:
            tick += 1
            d_move_order = (dire_act[0] * 1000,dire_act[1] * 1000)
            r_move_order = (rad_act[0] * 1000,rad_act[1] * 1000)
            _engine.set_move_order("Dire",0,dire_act[0] * 1000,dire_act[1] * 1000)
            _engine.set_move_order("Radiant",0,rad_act[0] * 1000,rad_act[1] * 1000)

            _engine.loop()
            if tick % param.tick_per_action != 0:
                continue#for faster training
            if canvas != None:
                #_engine.draw()
                canvas.update_idletasks()

            d_tup = _engine.get_state_tup("Dire", 0)
            r_tup = _engine.get_state_tup("Radiant", 0)

            #print("origin output ", d_tup , r_tup,flush=True)

            r_total_reward += r_tup[1]
            d_total_reward += d_tup[1]

            r_tup = (r_tup[0],r_tup[1] + dotproduct(p_rad_act,rad_act,1),r_tup[2])
            d_tup = (d_tup[0],d_tup[1] + dotproduct(p_dire_act,dire_act,1),d_tup[2])
               
            dire_act = get_action(dire_agent.step(d_tup))
            rad_act = get_action(rad_agent.step(r_tup))

            p_dire_act = _engine.predefined_step("Dire",0)
            p_rad_act = _engine.predefined_step("Radiant",0)

            #print(d_tup,r_tup)

            print("game %d t=%f,r_act=%s,r_reward=%f,d_act=%s,d_reward=%f"\
                %(count, _engine.get_time(),str(rad_act),r_tup[1],str(dire_act),d_tup[1]))
            
            last_dire_location = hero_location_by_tup(d_tup)
            last_rad_location = hero_location_by_tup(r_tup)

            yield

            if d_tup[2] or r_tup[2]:
                break
        print("total reward %f %f"%(r_total_reward, d_total_reward))
        rad_agent.fill_memory()
        dire_agent.fill_memory()

        if count > 0:
            for it in range(1):
                shared_grad_buffers = rad_agent.shared_grad_buffers
                start_t = time.time()
                rad_agent.train()
                dire_agent.train()
                t1 = time.time()
                print("trianing x2 : %fs"%(t1 - start_t))

                num_iter = num_iter + 1
                optimizer.zero_grad()

                for n,p in shared_model.named_parameters():
                    p._grad = Variable(shared_grad_buffers.grads[n+'_grad'])
                    p.data -= param.lr * p.grad.data
                
                #optimizer.step()
                shared_grad_buffers.reset()
                print("opt time: %fs"%(time.time() - t1))
                
                

            torch.save(shared_model.state_dict(),"./model/%d"%int(count))
            print('update')
            rad_agent.memory.clear()
            dire_agent.memory.clear()

if __name__ == '__main__':
    #1 process per cpu
    os.environ['OMP_NUM_THREADS'] = '1'
    num_processes = os.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("action",help = "start_server or simulator")
    parser.add_argument("--model",help = "path to pretrained model")
    parser.add_argument("-i",help = "start iteration")
    parser.add_argument("-np",help = "num of process")
    args = parser.parse_args()

    if args.np != None:
        num_processes = args.np

    params = Params()
    torch.manual_seed(params.seed)

    traffic_light = TrafficLight()
    counter = Counter()

    shared_model = Model(params.num_inputs, params.num_outputs)

    num_iter = 0

    if args.model != None:
        shared_model.load_state_dict(torch.load(args.model))
        print("loaded %s"%args.model)
    
    if args.i != None:
        num_iter = int(args.i)

    shared_model.share_memory()
    shared_grad_buffers = Shared_grad_buffers(shared_model)
    optimizer = optim.Adam(shared_model.parameters(), lr=params.lr)
    test_n = torch.Tensor([0])
    test_n.share_memory_()

    atomic_counter = AtomicInteger()
    CommonConV = threading.Barrier(3)

    rad_trainer = trainer(params,
    shared_model,shared_grad_buffers,None,
    atomic_counter,CommonConV)

    dire_trainer = trainer(params,
    shared_model,shared_grad_buffers,None,
    atomic_counter,CommonConV)

    dispatch_table["Radiant"] = rad_trainer
    dispatch_table["Dire"] = dire_trainer

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
    elif args.action == "cppSimulator_viz":
        g = start_cppSimulator()
        g.send(None)
        visualize(g, num_iter / params.num_epoch)
    elif args.action == "cppSimulator":
        g = start_cppSimulator()
        g.send(None)
        g.send(None)
        g.send((num_iter / params.num_epoch, None))
        while True:
            g.send(None)
    elif args.action == "mp_sim":
        mp_trainer(num_processes,shared_model,shared_grad_buffers,optimizer,it_num = num_iter)
    else:
        print("unknow action exit")
