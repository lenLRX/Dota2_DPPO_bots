import argparse
import os
import sys
import time
import threading
import _thread

from cppSimulator.cppSimulator import *
from visualizer import visualize

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
from mp_trainer import mp_trainer
from GameServer import start_env

try:
    from simulator.visualizer import visualize
except Exception:
    print("running without tkinter")
finally:
    pass

def start_cppSimulator():
    time.sleep(0.5)

    params,shared_model,shared_grad_buffers = yield

    num_iter, canvas = yield

    count = num_iter

    discount_factor = 1.0

    while True:
        _engine = cppSimulator(canvas)
        count += 1
        print("%d simulated game starts!"%count)
        dire_act = (0,None)
        rad_act = (0,None)
        dire_agent = trainer(params,shared_model,shared_grad_buffers)
        rad_agent = trainer(params,shared_model,shared_grad_buffers)

        shared_model = dire_agent.shared_model

        d_tup = _engine.get_state_tup("Dire", 0)
        r_tup = _engine.get_state_tup("Radiant", 0)

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
            #print(dire_act)

            _engine.loop()
            d_tup = _engine.get_state_tup("Dire", 0)
            r_tup = _engine.get_state_tup("Radiant", 0)

            if tick % param.tick_per_action != 0 and not(d_tup[2] or r_tup[2]):
                continue#for faster training
            if canvas != None:
                canvas.update_idletasks()

            #print("origin output ", d_tup , r_tup,flush=True)

            r_total_reward += r_tup[1]
            d_total_reward += d_tup[1]

            #dire_act = get_action(dire_agent.step(d_tup,p_dire_act,0))
            #rad_act = get_action(rad_agent.step(r_tup,p_rad_act,0))
            p_dire_act = _engine.predefined_step("Dire",0)
            p_rad_act = _engine.predefined_step("Radiant",0)
            
            dire_act = dire_agent.step(d_tup,p_dire_act,1)
            rad_act = rad_agent.step(r_tup,p_rad_act,1)

            

            #print(d_tup,r_tup)

            #print("game %d t=%f,r_act=%s,r_reward=%f,d_act=%s,d_reward=%f"\
            #    %(count, _engine.get_time(),str(rad_act),r_tup[1],str(dire_act),d_tup[1]))
            _engine.set_order("Dire",0,dire_act)
            _engine.set_order("Radiant",0,rad_act)

            yield

            if d_tup[2] or r_tup[2]:
                break
        print("total reward %f %f"%(r_total_reward, d_total_reward))

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
        num_processes = int(args.np)

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

    if args.action == "start_server":
        start_env(params,shared_model,shared_grad_buffers)
    elif args.action == "cppSimulator_viz":
        g = start_cppSimulator()
        g.send(None)
        g.send((params,shared_model,shared_grad_buffers))
        visualize(g, num_iter / params.num_epoch)
    elif args.action == "cppSimulator":
        g = start_cppSimulator()
        g.send(None)
        g.send((params,shared_model,shared_grad_buffers))
        g.send((0,None))
        while True:
            g.send(None)
    elif args.action == "mp_sim":
        mp_trainer(num_processes,shared_model,shared_grad_buffers,optimizer,it_num = num_iter)
    else:
        print("unknow action exit")
