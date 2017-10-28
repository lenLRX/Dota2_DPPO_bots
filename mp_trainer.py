import sys
import os
import time
import math

import torch.multiprocessing as mp
from torch.autograd import Variable

import numpy as np

from cppSimulator.cppSimulator import *
from train import trainer
from utils import *

def mp_trainer(np, model, grad_buffer, optimizer, it_num = 0):
    if np is None:
        print("can not get num of process!")
        sys.exit(-1)
    
    #np trainers and an optmizer
    Barrier = mp.Barrier(np + 1)
    Condition = mp.Condition()

    p_opt_args = (np,it_num,Barrier,optimizer,Condition,model,grad_buffer)
    p_opt = mp.Process(target = optimizer_process,args = p_opt_args)
    p_opt.start()

    processes = []
    processes.append(p_opt)

    for id in range(np):
        p_trainer_args = (id,it_num,Barrier,optimizer,Condition,model,grad_buffer)
        p_trainer = mp.Process(target = trainer_process, args = p_trainer_args)
        p_trainer.start()
        processes.append(p_trainer)
    
    for p in processes:
        p.join()


def optimizer_process(np,num,barrier,optimizer,condition,shared_model,shared_grad_buffers):
    #optimizer may use all cpus because all other process are waiting
    os.environ['OMP_NUM_THREADS'] = str(np)

    while True:
        num += 1
        barrier.wait()
        for n,p in shared_model.named_parameters():
            p._grad = Variable(shared_grad_buffers.grads[n+'_grad'])
        optimizer.step()
        shared_grad_buffers.reset()

        print("optimized %d"%num)
        print("log_std:",shared_model.state_dict()["log_std"])
        shared_model.state_dict()["log_std"] = torch.clamp(shared_model.state_dict()["log_std"], max=0)
        torch.save(shared_model.state_dict(),"./model/%d"%int(num))
        barrier.wait()

def trainer_process(id,num,barrier,optimizer,condition,shared_model,shared_grad_buffers):
    params = Params()
    canvas = None
    count = 0
    discount_factor = 100.0
    while True:
        count += 1
        _engine = cppSimulator(canvas)
        if id == 0:
            print("%d simulated game starts!"%count)

        dire_act = np.asarray([0.0,0.0])
        rad_act = np.asarray([0.0,0.0])
        dire_agent = trainer(params,shared_model,shared_grad_buffers)
        rad_agent = trainer(params,shared_model,shared_grad_buffers)

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

        while _engine.get_time() < 2000:
            tick += 1
            d_move_order = (dire_act[0] * 1000,dire_act[1] * 1000)
            r_move_order = (rad_act[0] * 1000,rad_act[1] * 1000)
            _engine.set_move_order("Dire",0,dire_act[0] * 1000,dire_act[1] * 1000)
            _engine.set_move_order("Radiant",0,rad_act[0] * 1000,rad_act[1] * 1000)

            _engine.loop()
            if tick %5 != 0:
                continue#for faster training

            if canvas != None:
                #_engine.draw()
                canvas.update_idletasks()

            d_tup = _engine.get_state_tup("Dire", 0)
            r_tup = _engine.get_state_tup("Radiant", 0)

            #print("origin output ", d_tup , r_tup,flush=True)

            r_total_reward += r_tup[1]
            d_total_reward += d_tup[1]
            

            dire_act = dire_agent.step(d_tup)
            rad_act = rad_agent.step(r_tup)

            #print("game %d t=%f,r_act=%s,r_reward=%f,d_act=%s,d_reward=%f"\
            #    %(count, _engine.get_time(),str(rad_act),r_tup[1],str(dire_act),d_tup[1]))
            
            last_dire_location = hero_location_by_tup(d_tup)
            last_rad_location = hero_location_by_tup(r_tup)

            if d_tup[2] or r_tup[2]:
                break
        if id == 0:
            print("total reward %f %f"%(r_total_reward, d_total_reward))
        rad_agent.fill_memory()
        dire_agent.fill_memory()

        if count > 0:
            avg_loss = ""
            for it in range(Params().num_epoch):
                start_t = time.time()
                avg_loss += rad_agent.train()
                avg_loss += dire_agent.train()
                t1 = time.time()
                if id == 0:
                    print("trianing x2 : %fs"%(t1 - start_t))
            
            if id == 0:
                print("loss %s"%avg_loss)
            rad_agent.memory.clear()
            dire_agent.memory.clear()

        barrier.wait()
        #training finished and wait for optimizer
        barrier.wait()
