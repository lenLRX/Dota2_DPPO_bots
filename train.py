import os
import sys
import numpy as np
import random
import math
import time
import threading

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp

from model import Model,_s_Shared_obs_stats

class ReplayMemory(object):
    def __init__(self, params, capacity):
        self.params = params
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        samples = zip(*random.sample(self.memory, batch_size))
        _out = []
        first = True
        for _s in samples:
            if first == True:
                first = False
                _out_dict = {}
                for k in self.params.num_inputs:
                    _out_dict[k] = []
                for _t in _s:
                    for k in _t:
                        if not k == "self_input":
                            _out_dict[k].append(_t[k])
                        else:
                            if isinstance(_t[k],list):
                                print(_s,k)
                            _out_dict[k].append(_t[k].view(
                                -1,self.params.num_inputs[k]))
                for k in _out_dict:
                    if not isinstance(_out_dict[k][0],list):
                        #dont cat list
                        _out_dict[k] = torch.cat(_out_dict[k],0)
                    #print(_out_dict[k].size())
                _out.append(_out_dict)
            else:
                #print(_s)
                _out.append(torch.cat(_s,0))
                #print(torch.cat(_s,0).size())
        return _out

    '''
    def determined_sample(self,x,batch_size,seed):
        random.seed(seed)
        return random.sample(x,batch_size)

    def sample(self,batch_size):
        seed = int(time.time())


    def prepare(self):
        self.datas = list(zip(*self.memory))
        self.
    '''

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            pass
        shared_param._grad = param.grad

def normal(x, mu, std):
    a = (-1*(x-mu).pow(2)/(2*std)).exp()
    b = 1/(2*std*np.pi).sqrt()
    return a*b

def detach_state(state):
    new_state = {}
    for k in state:
        if not k == "self_input":
            #print(state[k])
            _temp = []
            for t in state[k]:
                _temp.append(t[0].detach())
            new_state[k] = _temp
        else:
            new_state[k] = state[k].detach()
    
    return new_state


class trainer(object):
    def __init__(self,params, shared_model, shared_grad_buffers, shared_obs_stats, atomicInt, ChiefConV):
        self.atomicInt = atomicInt
        self.params = params
        torch.manual_seed(self.params.seed)
        self.model = Model(self.params.num_inputs,self.params.num_outputs)

        self.memory = ReplayMemory(params,10000)

        self.init_state = {"self_input":[0 for x in range(self.params.num_inputs["self_input"])],
                "ally_input":[[0 for x in range(self.params.num_inputs["ally_input"])]]}
        
        self.state = self.init_state

        self.done = False
        self.episode_length = 0
        self.shared_model = shared_model
        self.shared_obs_stats = shared_obs_stats
        self.shared_grad_buffers = shared_grad_buffers

        self.get_action_ConV = threading.Condition()
        self.get_state_Conv = threading.Condition()

        self.is_training = False

        self.ChiefConV = ChiefConV

        self.action_out = [0,0]
        self.state_tuple_in = None
        self.flag = False
        self.action_spin_flag = False

        self.has_last_action = False

        self.reward_normalizer = _s_Shared_obs_stats(1)
    
    def notifyTrainingFinish(self):
        while not self.is_training:
            time.sleep(1)
        self.is_training = False
        print("notify training finish")
    
    def waitTraningFinish(self):
        print("waiting training finish")
        self.is_training = True
        while self.is_training:
            time.sleep(1)
        print("waiting training finish done")
    
    def set_state(self,st):
        raise NotImplementedError()
        #self.get_state_Conv.acquire()
        self.state_tuple_in = (st["state"],float(st["reward"]),st["done"] == "true")
        #self.get_state_Conv.notify()
        #self.get_state_Conv.release()
        self.flag = True
        
    
    def get_action(self):
        raise NotImplementedError()
        #self.get_action_ConV.acquire()
        time.sleep(0.5)
        
        ret = self.action_out
        self.action_spin_flag = False
        #self.get_action_ConV.notify()
        #self.get_action_ConV.release()
        return ret

    def step(self,state_tuple_in):
        self.w += 1
        self.flag = True

        self.state, self.reward, self.done = state_tuple_in
        raw_reward = self.reward
        #_start_time = time.time()

        self.shared_obs_stats.observes(self.state)
        self.state = self.shared_obs_stats.normalize(self.state)
        
        mu, sigma_sq, v = self.model(self.state)
        eps = torch.randn(mu.size())

        if np.random.rand() < 0.05:
            self.action = Variable(torch.FloatTensor(np.random.rand(2) * 2 -1)).view(-1,2)
        else:
            self.action = (mu + sigma_sq.sqrt()*Variable(eps))
        

        self.action_out = self.action.data.squeeze().numpy()
        self.action_spin_flag = True
        #print(time.time() - _start_time)

        self.cum_reward += self.reward

        if self.has_last_action:
            self.states.append(self.state)
            self.actions.append(self.last_action)
            self.values.append(v)
            self.rewards.append(self.reward)
        
        self.last_action = self.action
        self.has_last_action = True
        
        #print("%d: action = %f %f value=%f reward = %f"%(
        #    self.w,self.action_out[0],self.action_out[1],float(v.data.numpy()[0]),raw_reward),sigma_sq.sqrt())
        
        return self.action_out


    def loop(self):
        while True:
            self.episode_length += 1
            self.model.load_state_dict(self.shared_model.state_dict())
        
            self.w = -1
            self.av_reward = 0
            self.nb_runs = 0
            self.reward_0 = 0
            self.t = -1
            while self.w < self.params.exploration_size:
                self.state = self.init_state
                print("explore %d"%self.w)
                self.t += 1
                self.states = []
                self.actions = []
                self.rewards = []
                self.values = []
                self.returns = []
                self.advantages = []
                self.av_reward = 0
                self.cum_reward = 0
                self.cum_done = 0

                _start_time = time.time()
                while time.time() - _start_time < 5:
                    if self.flag:
                        _start_time = time.time()
                        self.flag = False
                
                self.cum_done += 1
                self.av_reward += self.cum_reward
                self.cum_reward = 0
                self.episode_length = 0

                self.state_tuple_in = None
                self.done = False
                if len(self.states) < 50:
                    continue
                R = torch.zeros(1, 1)
                if not self.done:
                    _,_,v = self.model(self.state)
                R = v.data
                self.values.append(Variable(R))
                R = Variable(R)
                A = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(self.rewards))):
                    td = self.rewards[i] + self.params.gamma*self.values[i+1].view(-1).data[0] - self.values[i].view(-1).data[0]
                    A = td + self.params.gamma*self.params.gae_param*A
                    self.advantages.insert(0, A)
                    R = A + self.values[i]
                    self.returns.insert(0, R)
                # store usefull info:
                self.memory.push([self.states, self.actions, self.returns, self.advantages])
            # policy grad updates:
            self.av_reward /= float(self.cum_done + 1)
            model_old = Model(self.params.num_inputs, self.params.num_outputs)
            model_old.load_state_dict(self.model.state_dict())
            if self.t==0:
                self.reward_0 = self.av_reward-(1e-2)
            
            for k in range(self.params.num_epoch):
                # load new model
                self.model.load_state_dict(self.shared_model.state_dict())
                self.model.zero_grad()
                # new mini_batch
                batch_states, batch_actions, batch_returns, batch_advantages = self.memory.sample(self.params.batch_size)
                # old probas
                mu_old, sigma_sq_old, v_pred_old = model_old(detach_state(batch_states))
                probs_old = normal(batch_actions, mu_old, sigma_sq_old)
                # new probas
                mu, sigma_sq, v_pred = self.model(batch_states)
                probs = normal(batch_actions, mu, sigma_sq)
                # ratio
                ratio = probs/(1e-10+probs_old)
                # clip loss
                surr1 = ratio * torch.cat([batch_advantages]*self.params.num_outputs,1) # surrogate from conservative policy iteration
                surr2 = ratio.clamp(1-self.params.clip, 1 + self.params.clip) * torch.cat([batch_advantages]*self.params.num_outputs,1)
                loss_clip = -torch.mean(torch.min(surr1, surr2))
                # value loss
                vfloss1 = (v_pred - batch_returns)**2
                v_pred_clipped = v_pred_old + (v_pred - v_pred_old).clamp(-self.params.clip, self.params.clip)
                vfloss2 = (v_pred_clipped - batch_returns)
                loss_value = 0.5*torch.mean(torch.max(vfloss1, vfloss2))
                # entropy
                loss_ent = -self.params.ent_coeff*torch.mean(probs*torch.log(probs+1e-5))
                # total
                total_loss = (loss_clip + loss_value + loss_ent)
                print("training %d / %d loss = %f"%(k,self.params.num_epoch,total_loss.data[0]),torch.mean(batch_returns,0))
                # before step, update old_model:
                model_old.load_state_dict(self.model.state_dict())
                # prepare for step
                total_loss.backward(retain_variables=True)
                #ensure_shared_grads(model, shared_model)
                #shared_model.cum_grads()
                self.shared_grad_buffers.add_gradient(self.model)

                self.atomicInt.inc()

                self.ChiefConV.wait()

            self.memory.clear()
            self.notifyTrainingFinish()