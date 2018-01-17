import sys
import os
import time
import math

import torch.multiprocessing as mp
from torch.autograd import Variable

import numpy as np

from SimDota2 import *
from train import trainer
from utils import *

def mp_trainer(np, batch_size, model, grad_buffer, optimizer, it_num = 0):
    if np is None:
        print("can not get num of process!")
        sys.exit(-1)
    
    #np trainers and an optmizer
    Barrier = mp.Barrier(batch_size + 1)
    Condition = mp.Condition()
    Semaphore = mp.Semaphore(np)

    shared_score = torch.FloatTensor([0])
    shared_score.share_memory_()

    p_opt_args = (np,batch_size,it_num,Barrier,optimizer,Condition,model,grad_buffer,shared_score)
    p_opt = mp.Process(target = optimizer_process,args = p_opt_args)
    p_opt.start()

    processes = []
    processes.append(p_opt)

    

    for id in range(batch_size):
        p_trainer_args = (id,it_num,Barrier,Semaphore,optimizer,Condition,model,grad_buffer,shared_score,np)
        p_trainer = mp.Process(target = trainer_process, args = p_trainer_args)
        p_trainer.start()
        processes.append(p_trainer)
    
    for p in processes:
        p.join()


def optimizer_process(np,batch_size,num,barrier,optimizer,condition,shared_model,shared_grad_buffers,shared_score):
    #optimizer may use all cpus because all other process are waiting
    os.environ['OMP_NUM_THREADS'] = str(np)

    while True:
        num += 1
        barrier.wait()
        bHoldon = False
        if not bHoldon:
            for n,p in shared_model.named_parameters():
                p._grad = Variable(shared_grad_buffers.grads[n+'_grad'])
                p._grad.data.clamp_(-param.grad_clip,param.grad_clip)
                delta = param.lr * p._grad.data / float(batch_size)
                #print(n,delta,"_grad",p._grad)
                p.data -= delta
            #optimizer.step()
            shared_grad_buffers.reset()

            shared_score[0] = shared_score[0] / batch_size
            print("optimized %d total reward %f"%(num,shared_score[0]))
            shared_score[0] = 0
            if num % 100:
                torch.save(shared_model.state_dict(),"./model/%d"%int(num))
        barrier.wait()

def trainer_process(id,num,barrier,semaphore,optimizer,condition,shared_model,shared_grad_buffers,shared_score,num_process):
    randst = np.random.mtrand.RandomState(os.getpid())
    params = Params()
    canvas = None
    count = 0
    discount_factor = 1
    while True:
        semaphore.acquire()
        count += 1
        _engine = Simulator(canvas)
        if id == 0:
            print("%d simulated game starts!"%count)

        dire_act = (0,None)
        rad_act = (0,None)
        dire_agent = trainer(params,shared_model,shared_grad_buffers)
        rad_agent = trainer(params,shared_model,shared_grad_buffers)

        d_tup = _engine.get_state_tup("Dire", 0)
        r_tup = _engine.get_state_tup("Radiant", 0)

        r_total_reward = 0.0
        d_total_reward = 0.0

        discount_factor = randst.rand()
        print(id,discount_factor)
        _flag = 0
        if discount_factor < 0.5:
            _flag = 1
        
        dire_agent.pre_train()
        rad_agent.pre_train()

        if discount_factor < 0.0:
            discount_factor = 0.0
        
        tick = 0

        p_dire_act = _engine.predefined_step("Dire",0)
        p_rad_act = _engine.predefined_step("Radiant",0)

        while _engine.get_time() < param.game_duriation:
            tick += 1

            _engine.loop()
            d_tup = _engine.get_state_tup("Dire", 0)
            r_tup = _engine.get_state_tup("Radiant", 0)

            if tick % param.tick_per_action != 0 and not(d_tup[2] or r_tup[2]):
                continue#for faster training
            if canvas != None:
                #_engine.draw()
                canvas.update_idletasks()

            

            #print("origin output ", d_tup , r_tup,flush=True)

            r_total_reward += r_tup[1]
            d_total_reward += d_tup[1]

            #r_tup = (r_tup[0],r_tup[1] + 0.1 * dotproduct(p_rad_act,rad_act,1),r_tup[2])
            #d_tup = (d_tup[0],d_tup[1] + 0.1 * dotproduct(p_dire_act,dire_act,1),d_tup[2])

            #print("game %d t=%f,r_act=%s,%s,r_reward=%f,d_act=%s %s,d_reward=%f"\
            #    %(count, _engine.get_time(),str(rad_act),str(p_rad_act),r_tup[1],str(dire_act),str(p_dire_act),d_tup[1]))
            #r_tup = (r_tup[0],r_tup[1] - 0.01,r_tup[2])
            #d_tup = (d_tup[0],d_tup[1] - 0.01,d_tup[2])
               
            p_dire_act = _engine.predefined_step("Dire",0)
            p_rad_act = _engine.predefined_step("Radiant",0)
            
            dire_act = dire_agent.step(d_tup,p_dire_act,_flag)
            rad_act = rad_agent.step(r_tup,p_rad_act,_flag)

            _engine.set_order("Dire",0,dire_act)
            _engine.set_order("Radiant",0,rad_act)

            #print(d_tup,r_tup)

            if d_tup[2] or r_tup[2]:
                break
        
        shared_score[0] += (r_total_reward + d_total_reward) * 0.5

        if count > 0:
            avg_loss = 0.0
            for it in range(Params().num_epoch):
                start_t = time.time()
                bHoldon = False
                avg_loss += rad_agent.train(bHoldon)
                avg_loss += dire_agent.train(bHoldon)
                t1 = time.time()
                if id == 0:
                    print("trianing x2 : %fs"%(t1 - start_t))
            avg_loss *= 0.5
            if id == 0:
                print("loss %s"%str(avg_loss))
            rad_agent.memory.clear()
            dire_agent.memory.clear()

        semaphore.release()
        barrier.wait()
        #training finished and wait for optimizer
        barrier.wait()
