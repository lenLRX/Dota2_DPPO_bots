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
        #samples = zip(*random.sample(self.memory, batch_size))
        samples = zip(*self.memory)
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

def get_nearest_act(pd):
    a = math.atan2(*pd)
    _min = 10000
    _idx = -1
    for i in range(param.num_outputs ** 2):
        _d = abs(p_acts[i] - a)
        if _d < _min:
            _min = _d
            _idx = i
    return _idx

class trainer(object):
    def __init__(self,params, shared_model, shared_grad_buffers,
        shared_obs_stats = None):
        self.params = params
        torch.manual_seed(int(time.time()))
        self.model = Model(self.params.num_inputs,self.params.num_outputs)
        self.model.load_state_dict(shared_model.state_dict())

        self.memory = ReplayMemory(params,10000000)

        self.done = False
        self.episode_length = 0
        self.shared_model = shared_model
        self.shared_grad_buffers = shared_grad_buffers


        self.is_training = False

        self.state_tuple_in = None
        self.flag = False

        self.has_last_action = False

        self.first_print = True

    def pre_train(self):
        self.decisions = []
        self.decisions_log = []
        self.subdecisions = []
        self.subdecisions_log = []
        self.predefined_actions = []#calculate reward
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []

    def step(self, state_tuple_in,predefine = None, c = 0.2):
        #_start_time = time.time()

        self.state, self.reward, self.done = state_tuple_in

        v_out, decision, decision_layer_log_out,\
                move_target, move_target_log,\
                atk_target, atk_target_log = self.model(self.state)
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
        '''
        bUsePredefine = False
        if np.random.rand() < c:
            self.action,self.subaction = predefine
            if self.action == 1:
                #print(self.subaction)
                self.subaction = get_nearest_act(self.subaction)
            bUsePredefine = True
            
        self.action_log = decision_layer_log_out
        if 0 == decision:
            #noop
            if not bUsePredefine:
                self.action = 0
                self.subaction = None
            self.subaction_log = None
        elif 1 == decision:
            #move
            if not bUsePredefine:
                self.action = 1
                self.subaction = move_target
            self.subaction_log = move_target_log
        elif 2 == decision:
            #attack
            if not bUsePredefine:
                self.action = 2
                self.subaction = atk_target
            self.subaction_log = atk_target_log
            if self.subaction is None:
                self.action = 0
                self.subaction_log = None
        
        if self.has_last_action:
            self.decisions.append(self.last_action)
            self.decisions_log.append(self.last_action_log)
            self.subdecisions.append(self.last_subaction)
            self.subdecisions_log.append(self.last_subaction_log)
            self.predefined_actions.append(self.last_predefine_action)

            self.values.append(v_out)

            self.rewards.append(self.reward)

        if self.first_print:
            print("act",self.action,"value",v_out,"action",decision_layer_log_out)
            self.first_print = False
        
        
        self.last_action = self.action
        self.last_action_log = self.action_log
        self.last_subaction = self.subaction
        self.last_subaction_log = self.subaction_log
        self.last_predefine_action = predefine
        self.has_last_action = True

        if self.action == 1:
            return (self.action, get_action(self.subaction))
        else:
            return (self.action,self.subaction)

    def train(self):
        def equal_to_predefine(self, act, idx):
            return not self.predefined_actions[idx] is None and self.predefined_actions[idx] == act

        self.model.zero_grad()
        R = torch.zeros(1, 1)
        self.values.append(self.values[-1])
        R = Variable(R)
        A = Variable(torch.zeros(1, 1))
        loss = Variable(torch.FloatTensor([0.0]))
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.params.gamma*R
            A = R - self.values[i].view(-1).data[0]
            additional_reward = 0.0
            decision = self.decisions[i]
            subdecision_policy_loss = 0.0
            if 0 == decision:
                pass
            elif 1 == decision:
                #move
                if equal_to_predefine(self, 1, i):
                    additional_reward = additional_reward + dotproduct(self.predefined_actions[i][1],get_action(self.subdecisions[i]),1)
                #if not self.subdecisions_log[i] is None:
                    _log = Variable(torch.zeros(1, self.subdecisions_log[i].view(-1).size()[0]))
                    try:
                        _log.data[0][self.subdecisions[i]] = 1
                    except Exception as e:
                        print(_log,self.subdecisions[i],decision,i)
                        raise
                    _log = _log * self.subdecisions_log[i]
                    subdecision_policy_loss = - ((A + additional_reward) * _log).view(-1)
                    subdecision_policy_loss = torch.sum(subdecision_policy_loss)
            elif 2 == decision:
                #atk
                if not self.subdecisions[i] is None:
                    #invalid decision,punish decision
                    additional_reward = -param.atk_addtion_rwd
                else:
                    if equal_to_predefine(self, 2, i):
                        #predefined idx
                        if self.predefined_actions[i] == self.subaction[i]: 
                            additional_reward = param.atk_addtion_rwd
                        else:
                            additional_reward = -param.atk_addtion_rwd
                    #if not self.subdecisions_log[i] is None:
                        _log = Variable(torch.zeros(1, self.subdecisions_log[i].view(-1).size()[0]))
                        _log.data[0][self.subdecisions] = 1
                        _log = _log * self.subdecisions_log[i]
                        subdecision_policy_loss = - ((A + additional_reward) * _log).view(-1)
                        subdecision_policy_loss = torch.sum(subdecision_policy_loss)
            
            value_loss = (R + additional_reward - self.values[i].view(-1)) ** 2
            decision_policy_loss = - ((A + additional_reward) * self.decisions_log[i]).view(-1)
            decision_policy_loss = torch.mean(decision_policy_loss)

            loss = loss + decision_policy_loss + 0.5 * value_loss + subdecision_policy_loss
        loss = loss / len(self.rewards)
        loss.backward()
        self.shared_grad_buffers.add_gradient(self.model)
        return float(loss.data.numpy()[0])



    def fill_memory(self):
        
        R = torch.zeros(1, 1)
        self.values.append(self.values[-1])
        R = Variable(R)
        A = Variable(torch.zeros(1, 1))
        
        for i in reversed(range(len(self.rewards))):
            try:
                #td = self.rewards[i] + self.params.gamma*self.values[i+1].view(-1) - self.values[i].view(-1)
                #A = td + self.params.gamma*self.params.gae_param*A
                R = self.rewards[i] + self.params.gamma*R
                A = R - self.values[i].view(-1).data[0]
                additional_reward = 0.0
                if not self.predefined_actions[i] is None:
                    additional_reward = additional_reward + dotproduct(self.predefined_actions[i],self.lattice_actions[i],1)
                #print(additional_reward)
                self.advantages.insert(0, A + additional_reward)
                #R = self.rewards[i] + self.params.gamma*R 
                self.returns.insert(0, R + additional_reward)
            except Exception as e:
                print(str(e))
                print("error at %d"%i)
                with open("./debug.out", "w+") as dbgout:
                    for r in self.rewards:
                        dbgout.write("%d  %s\n"%(i,str(r)))
                    dbgout.write("\n\n\n\n")
                    for v in self.values:
                        dbgout.write("%d  %s\n"%(i,str(v)))
        
        #print("rewards",self.rewards)
        #print("returns",self.returns)
        # store usefull info:
        #print(type(self.actions))
        self.memory.push([self.states, self.actions, self.returns, 
            self.advantages, self.values])
        #raise "stop"
    
    def train_(self):
        self.model.zero_grad()

        # new mini_batch
        batch_log_actions, batch_returns, batch_advantages, batch_values = self.memory.sample(self.params.batch_size)

        #print("adv",batch_advantages)
        #print("batch_values",batch_values)
        #print("actions",batch_log_actions)

        policy_loss = -torch.sum(batch_log_actions * batch_advantages,1).view(-1)
        policy_loss = torch.mean(policy_loss)
        #print("policy_loss",policy_loss)
        value_loss = torch.sum((batch_returns - batch_values) ** 2,1).view(-1)
        value_loss = torch.mean(value_loss)
        #print("value_loss",value_loss)
        #print("batch_return",(batch_returns))

        total_loss = policy_loss + 0.5 * value_loss
        #print("training  loss = ", total_loss, torch.mean(batch_returns,0))
        # prepare for step
        total_loss.backward()
        #ensure_shared_grads(model, shared_model)
        #shared_model.cum_grads()
        self.shared_grad_buffers.add_gradient(self.model)
        return str(total_loss)
