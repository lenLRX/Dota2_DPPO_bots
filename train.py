import os
import sys
import numpy as np
import random
import math
import time
import threading
import code

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp

from model import Model
from utils import *

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
            _out.append(torch.cat(_s,0))
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

p_acts = []

for i in range(param.num_outputs ** 2):
    _act = get_action(i)
    if _act[0] != 0 and _act[1] != 0:
        p_acts.append(math.atan2(*_act))
    else:
        p_acts.append(100000)

class trainer(object):
    def __init__(self,params, shared_model, shared_grad_buffers,
        shared_obs_stats = None, atomicInt = None, ChiefConV = None):
        self.atomicInt = atomicInt
        self.params = params
        torch.manual_seed(self.params.seed)
        self.model = Model(self.params.num_inputs,self.params.num_outputs)
        self.model.load_state_dict(shared_model.state_dict())

        self.memory = ReplayMemory(params,10000000)

        self.init_state = {"self_input":[0 for x in range(self.params.num_inputs["self_input"])],
                "ally_input":[[0 for x in range(self.params.num_inputs["ally_input"])]]}
        
        self.state = self.init_state

        self.done = False
        self.episode_length = 0
        self.shared_model = shared_model
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

    def pre_train(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []
        
        self.av_reward = 0
        self.cum_reward = 0
        self.cum_done = 0

    def step(self, state_tuple_in,predefine = None, c = 0.2):
        #_start_time = time.time()

        self.state, self.reward, self.done = state_tuple_in

        self.state = Variable(torch.FloatTensor(self.state)).view(1,1,-1)

        s_action, v, log_action = self.model(self.state)
        print(s_action,v)

        '''
        if np.random.rand() < 0.05:
            self.action = Variable(torch.FloatTensor(np.random.rand(2) * 2 -1)).view(1,-1,2)
        else:
            self.action = (mu + sigma_sq.sqrt()*Variable(eps))
        '''
        if np.random.rand() < c:
            if predefine is None:
                self.action = np.random.choice(self.params.num_outputs**2)
            else:
                a = math.atan2(*predefine)
                _min = 10000
                for i in range(param.num_outputs ** 2):
                    _d = abs(p_acts[i] - a)
                    if _d < _min:
                        _min = _d
                        self.action = i
                #print(predefine,get_action(self.action))
                #self.action = int(predefine[0])
        else:
            self.action = np.argmax(s_action.data.numpy()[0])
            #self.action = np.random.choice(self.params.num_outputs**2,p = s_action.data.numpy()[0])
        #print(s_action)
        #self.action = np.random.choice(self.params.num_outputs**2,p = s_action.data.numpy()[0])

        self.cum_reward += self.reward

        
        if self.has_last_action:
            self.states.append(self.state)

            self.actions.append(self.last_log_action)
            self.values.append(v)

            self.rewards.append(self.reward)

        
        self.last_log_action = Variable(torch.zeros(1, self.params.num_outputs ** 2))
        self.last_log_action.data[0][self.action] = 1
        self.last_log_action = self.last_log_action * log_action
        self.has_last_action = True

        
        #print("%d: action = %f %f value=%f reward = %f"%(
        #    self.w,self.action_out[0],self.action_out[1],float(v.data.numpy()[0]),self.reward),sigma_sq.sqrt())
        #print("total time",time.time() - _start_time)
        return self.action

    def fill_memory(self):
        R = torch.zeros(1, 1)
        self.values.append(Variable(R))
        R = Variable(R)
        A = Variable(torch.zeros(1, 1))
        for i in reversed(range(len(self.rewards))):
            try:
                td = self.rewards[i] + self.params.gamma*self.values[i+1].view(-1).data[0] - self.values[i].view(-1).data[0]
                A = td + self.params.gamma*self.params.gae_param*A
                self.advantages.insert(0, A)
                R = A + self.values[i]
                self.returns.insert(0, R)
            except:
                print("error at %d"%i)
                with open("./debug.out", "w+") as dbgout:
                    for r in self.rewards:
                        dbgout.write("%d  %s\n"%(i,str(r)))
                    dbgout.write("\n\n\n\n")
                    for v in self.values:
                        dbgout.write("%d  %s\n"%(i,str(v)))

        # store usefull info:
        #print(type(self.actions))
        self.memory.push([self.states, self.actions, self.returns, 
            self.advantages, self.values])
        #raise "stop"
    
    def train(self):
        self.model.zero_grad()

        # new mini_batch
        batch_states, batch_log_actions, batch_returns, batch_advantages, batch_values = self.memory.sample(self.params.batch_size)

        #print(batch_advantages)

        policy_loss = - torch.sum(batch_log_actions * batch_advantages,1).view(-1)
        policy_loss = torch.sum(policy_loss)
        #print("policy_loss",policy_loss)
        value_loss = torch.sum((batch_returns - batch_values) ** 2,1).view(-1)
        value_loss = torch.sum(value_loss)
        #print("value_loss",value_loss)

        total_loss = policy_loss + 0.5 * value_loss
        #print("training  loss = ", total_loss, torch.mean(batch_returns,0))
        # prepare for step
        total_loss.backward(retain_variables=True)
        #ensure_shared_grads(model, shared_model)
        #shared_model.cum_grads()
        self.shared_grad_buffers.add_gradient(self.model)
        return str(total_loss)