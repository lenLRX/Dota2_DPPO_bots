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

    shared_score = torch.FloatTensor([0])
    shared_score.share_memory_()

    for id in range(np):
        p_trainer_args = (id,it_num,Barrier,optimizer,Condition,model,grad_buffer,shared_score,np)
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
        if num % param.game_per_update == 0:
            for n,p in shared_model.named_parameters():
                p._grad = Variable(shared_grad_buffers.grads[n+'_grad'])
                delta = param.lr * p.grad.data / float(param.game_per_update)
                print(n,delta,"_grad",p._grad)
                p.data -= delta
            #optimizer.step()
            shared_grad_buffers.reset()

            print("optimized %d"%num)
            torch.save(shared_model.state_dict(),"./model/%d"%int(num))
        barrier.wait()

def trainer_process(id,num,barrier,optimizer,condition,shared_model,shared_grad_buffers,shared_score,num_process):
    randst = np.random.mtrand.RandomState(os.getpid())
    params = Params()
    canvas = None
    count = 0
    discount_factor = 1
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

        discount_factor = randst.rand()
        print(id,discount_factor)

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

            r_tup = (r_tup[0],0*r_tup[1] + 0.1 * dotproduct(p_rad_act,rad_act,1),r_tup[2])
            d_tup = (d_tup[0],0*d_tup[1] + 0.1 * dotproduct(p_dire_act,dire_act,1),d_tup[2])

            print("game %d t=%f,r_act=%s,%s,r_reward=%f,d_act=%s %s,d_reward=%f"\
                %(count, _engine.get_time(),str(rad_act),str(p_rad_act),r_tup[1],str(dire_act),str(p_dire_act),d_tup[1]))
            #r_tup = (r_tup[0],r_tup[1] - 0.01,r_tup[2])
            #d_tup = (d_tup[0],d_tup[1] - 0.01,d_tup[2])
               
            dire_act = get_action(dire_agent.step(d_tup,p_dire_act,discount_factor))
            rad_act = get_action(rad_agent.step(r_tup,p_rad_act,discount_factor))
            
            p_dire_act = _engine.predefined_step("Dire",0)
            p_rad_act = _engine.predefined_step("Radiant",0)

            #print(d_tup,r_tup)

            
            last_dire_location = hero_location_by_tup(d_tup)
            last_rad_location = hero_location_by_tup(r_tup)

            if d_tup[2] or r_tup[2]:
                break
        
        shared_score[0] += (r_total_reward + d_total_reward) * 0.5

        if id == 0:
            shared_score[0] = shared_score[0] / num_process
            print("total reward %f"%(shared_score[0]))
            shared_score[0] = 0
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
