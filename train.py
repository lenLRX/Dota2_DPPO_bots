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


def equal_to_original_decision(self, act, idx):
    return self.original_decisions[idx] == act

def equal_to_predefined_action(self, act, idx):
    return  (not self.predefined_actions[idx] is None) and self.predefined_actions[idx][0] == act

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
        self.loss = Variable(torch.FloatTensor([0.0]))
        self.holdon_cnt = 0

    def pre_train(self):
        self.decisions = []
        self.original_decisions = []
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
            
        
        if self.has_last_action:
            self.decisions.append(self.last_action)
            self.decisions_log.append(self.last_action_log)
            self.subdecisions.append(self.last_subaction)
            self.subdecisions_log.append(self.last_subaction_log)
            self.predefined_actions.append(self.last_predefine_action)

            self.original_decisions.append(self.last_original_decision)

            self.values.append(v_out)

            self.rewards.append(self.reward)

        if self.first_print:
            print("act",self.action,"value",v_out,"action",decision_layer_log_out)
            self.first_print = False
        
        
        self.last_action = self.action
        self.last_original_decision = decision
        self.last_action_log = self.action_log
        self.last_subaction = self.subaction
        self.last_subaction_log = self.subaction_log
        self.last_predefine_action = predefine
        self.has_last_action = True

        if self.action == 1:
            return (self.action, get_action(self.subaction))
        elif self.action == 2 and self.subaction is None:
            return (0,None)
        else:
            return (self.action,self.subaction)

    def train(self, holdon = False):

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
            #the action we actual taken
            decision = self.decisions[i]
            subdecision_policy_loss = 0.0
            if 0 == decision:
                pass
            elif 1 == decision:
                #move
                #actual action is move AND original action is move
                if equal_to_original_decision(self, 1, i):
                    #AND predefined action is move
                    if equal_to_predefined_action(self, 1, i):
                        additional_reward = additional_reward + 0.1 * dotproduct(self.predefined_actions[i][1],get_action(self.subdecisions[i]),1)
                    #print(additional_reward)
                    _log = Variable(torch.zeros(1, self.subdecisions_log[i].view(-1).size()[0]))
                    try:
                        _log.data[0][self.subdecisions[i]] = 1
                    except Exception as e:
                        print(_log,self.subdecisions[i],decision,i)
                        raise
                    _log = _log * self.subdecisions_log[i]
                    subdecision_policy_loss = - ((A + additional_reward) * _log).view(-1)
                    #subdecision_policy_loss = - (A * _log).view(-1)
                    subdecision_policy_loss = torch.sum(subdecision_policy_loss)
            elif 2 == decision:
                #atk
                if not self.subdecisions[i] is None:
                    #invalid decision,punish decision
                    additional_reward = -param.atk_addtion_rwd
                else:
                    #actual action is attacking AND original action is attacking
                    if equal_to_original_decision(self, 2, i):
                        #AND predefined action is attacking
                        if equal_to_predefined_action(self, 2, i):
                            if self.predefined_actions[i][1] == self.subaction[i]: 
                                additional_reward = param.atk_addtion_rwd
                            else:
                                additional_reward = -param.atk_addtion_rwd
                    if not self.subdecisions_log[i] is None:
                        _log = Variable(torch.zeros(1, self.subdecisions_log[i].view(-1).size()[0]))
                        _log.data[0][self.subdecisions[i]] = 1
                        _log = _log * self.subdecisions_log[i]
                        subdecision_policy_loss = - ((A + additional_reward) * _log).view(-1)
                        #subdecision_policy_loss = - (A * _log).view(-1)
                        subdecision_policy_loss = torch.sum(subdecision_policy_loss)
            
            if not self.predefined_actions[i] is None:
                _p = self.predefined_actions[i][0]
                if _p != decision:
                    additional_reward = -0.02
                else:
                    additional_reward =  0.02

            _d_log = Variable(torch.zeros(1, self.decisions_log[i].view(-1).size()[0]))
            _d_log.data[0][self.decisions[i]] = 1
            _d_log = _d_log * self.decisions_log[i]
            value_loss = (R - self.values[i].view(-1)) ** 2
            decision_policy_loss = - ((A + additional_reward) * _d_log).view(-1)
            #decision_policy_loss = - (A * _d_log).view(-1)
            decision_policy_loss = torch.mean(decision_policy_loss)

            loss = loss + decision_policy_loss + 0.5 * value_loss + subdecision_policy_loss
        loss = loss / len(self.rewards)
        self.loss = self.loss + loss
        self.holdon_cnt = self.holdon_cnt + 1
        if not holdon:
            #hold on loss, do not update until holdon is False
            self.loss = self.loss / self.holdon_cnt
            self.loss.backward()
            self.shared_grad_buffers.add_gradient(self.model)
            self.loss = Variable(torch.FloatTensor([0.0]))
            self.holdon_cnt = 0
        else:
            self.shared_grad_buffers.reset()
        return float(self.loss.data.numpy()[0])