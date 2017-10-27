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

from model import Model,Shared_obs_stats,_s_Shared_obs_stats

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
        i = 0
        for _s in samples:
            i += 1
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
                            #print(_s,k)
                            _out_dict[k].append(_t[k].view(
                                -1,self.params.num_inputs[k]))
                for k in _out_dict:
                    if not isinstance(_out_dict[k][0],list):
                        #dont cat list
                        _out_dict[k] = torch.cat(_out_dict[k],0)
                    #print(_out_dict[k].size())
                _out.append(_out_dict)
            else:
                if i == 5:
                    _out.append((torch.cat(_h,0).detach() for _h in zip(*_s)))
                elif i < 5:
                    _out.append(torch.cat(_s,0))
                else:
                    _out.append(_s)
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
        self.shared_obs_stats = Shared_obs_stats(params.num_inputs)
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

        self.reset_ICM()
    
    def reset_ICM(self):
        self.raw_actions = []
        self.inverses = []
        self.forwards = []
        self.sts = []
        self.st1s = []
        self.state1s = []
    
    def ICM_loss(self,batch_actions, batch_state1s, batch_inverses, batch_forwards):
        inverse_loss = 0.0
        forward_loss = 0.0
        for i in range(len(batch_actions)):
            inverse_err = batch_inverses[i] - batch_actions[i]
            inverse_loss = inverse_loss + 0.5 * (inverse_err.pow(2)).sum(2)
            forward_err = batch_forwards[i] - batch_state1s[i]
            forward_loss = forward_loss + 0.5 * (forward_err.pow(2)).sum(2)
        
        return inverse_loss, forward_loss

    def pre_train(self):
        self.states = []
        self.hidden_state = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []
        
        self.av_reward = 0
        self.cum_reward = 0
        self.cum_done = 0
        self.model.init_lstm()

    def step(self, state_tuple_in):
        #_start_time = time.time()

        self.state, self.reward, self.done = state_tuple_in

        self.shared_obs_stats.observes(self.state)
        self.state = self.shared_obs_stats.normalize(self.state)

        
        #nn_start_time = time.time()
        mu, sigma_sq, v, lstm_out, lstm_hidden = self.model(False, self.state)
        
        eps = torch.randn(mu.size())
        #print(time.time() - nn_start_time)

        if np.random.rand() < 0.05:
            self.action = Variable(torch.FloatTensor(np.random.rand(2) * 2 -1)).view(1,-1,2)
        else:
            #self.action = (mu + sigma_sq.sqrt()*Variable(eps))
            self.action = mu
        
        self.raw_action = self.action
        

        self.action_out = self.action.data.squeeze().numpy()
        self.action_spin_flag = True
        #print(time.time() - _start_time)

        self.cum_reward += self.reward

        
        if self.has_last_action:
            self.states.append(self.state)
            self.hidden_state.append(self.model.lstm_hidden)
            #print(self.hidden_state)
            self.actions.append(self.last_action)
            self.raw_actions.append(self.last_raw_action)
            self.values.append(v)
            

            vec_st1, inverse, forward = self.model(True,(
                    self.last_st,
                    lstm_out,
                    self.last_action))

            reward_intrinsic = ((vec_st1 - forward).pow(2)).sum(2) * 0.5
            self.reward += float(reward_intrinsic.data.numpy()[0][0])
            self.rewards.append(self.reward)
            self.state1s.append(vec_st1)
            self.inverses.append(inverse)
            self.forwards.append(forward)

        
        self.last_action = self.action
        self.last_raw_action = self.raw_action
        self.last_st = lstm_out
        self.has_last_action = True
        
        #print("%d: action = %f %f value=%f reward = %f"%(
        #    self.w,self.action_out[0],self.action_out[1],float(v.data.numpy()[0]),self.reward),sigma_sq.sqrt())
        #print("total time",time.time() - _start_time)
        return self.action_out

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
        #print(self.hidden_state)
        self.memory.push([self.states, self.actions, self.returns, 
            self.advantages, self.hidden_state,
            #for ICM
            self.raw_actions, self.state1s,
            self.inverses, self.forwards])
    
    def train(self):
        model_old = Model(self.params.num_inputs, self.params.num_outputs)
        model_old.load_state_dict(self.model.state_dict())
        model_old.init_lstm()
        # load new model
        self.model.load_state_dict(self.shared_model.state_dict())
        self.model.zero_grad()
        # new mini_batch
        batch_states, batch_actions, batch_returns, batch_advantages, batch_hidden_state,\
            _actions, batch_state1s, batch_inverses, batch_forwards = self.memory.sample(self.params.batch_size)
        # old probas
        mu_old, sigma_sq_old, v_pred_old, _, _ = model_old(False,detach_state(batch_states),batch_hidden_state)
        probs_old = normal(batch_actions, mu_old, sigma_sq_old)
        # new probas
        mu, sigma_sq, v_pred, _, _ = self.model(False, batch_states)
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
        inverse_loss, forward_loss = self.ICM_loss(_actions, batch_state1s, batch_inverses, batch_forwards)
        self.reset_ICM()
        #print("inverse_loss, forward_loss",inverse_loss, forward_loss)
        total_loss = (loss_clip + loss_value + loss_ent + 0.005 * inverse_loss + 0.01 * forward_loss)
        print("training  loss = ", total_loss, torch.mean(batch_returns,0))
        # before step, update old_model:
        model_old.load_state_dict(self.model.state_dict())
        # prepare for step
        total_loss.backward(retain_variables=True)
        #ensure_shared_grads(model, shared_model)
        #shared_model.cum_grads()
        self.shared_grad_buffers.add_gradient(self.model)
        return str(total_loss)