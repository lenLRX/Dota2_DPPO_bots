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

from model import Model

class ReplayMemory(object):
    def __init__(self, capacity):
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
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            pass
        shared_param._grad = param.grad

def normal(x, mu, std):
    a = (-1*(x-mu).pow(2)/(2*std)).exp()
    b = 1/(2*std*np.pi).sqrt()
    return a*b

class trainer(object):
    def __init__(self,params, shared_model, shared_grad_buffers, shared_obs_stats, atomicInt, ChiefConV):
        self.atomicInt = atomicInt
        self.params = params
        torch.manual_seed(self.params.seed)
        self.model = Model(self.params.num_inputs,self.params.num_outputs)

        self.memory = ReplayMemory(self.params.exploration_size)

        self.state = Variable(torch.Tensor(np.zeros((self.params.num_inputs))).unsqueeze(0))
        self.done = True
        self.episode_length = 0
        self.shared_model = shared_model
        self.shared_obs_stats = shared_obs_stats
        self.shared_grad_buffers = shared_grad_buffers

        self.get_action_ConV = threading.Condition()
        self.get_state_Conv = threading.Condition()

        self.ChiefConV = ChiefConV

        self.action_out = None
        self.state_tuple_in = None
    
    def set_state(self,st):
        self.get_state_Conv.acquire()
        self.state_tuple_in = (st["state"]["self_input"],float(st["reward"]),st["first"] == "true")
        self.get_state_Conv.notify()
        self.get_state_Conv.release()
        
    
    def get_action(self):
        self.get_action_ConV.acquire()
        ret = self.action_out
        self.get_action_ConV.notify()
        self.get_action_ConV.release()
        return ret
    
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

                for step in range(self.params.num_steps):
                    self.w += 1
                    self.shared_obs_stats.observes(self.state)
                    self.state = self.shared_obs_stats.normalize(self.state)
                    self.states.append(self.state)
                    mu, sigma_sq, v = self.model(self.state)
                    eps = torch.randn(mu.size())
                    self.action = (mu + sigma_sq.sqrt()*Variable(eps))
                    self.actions.append(self.action)
                    self.values.append(v)

                    self.get_action_ConV.acquire()
                    self.action_out = self.action.data.squeeze().numpy()
                    self.get_action_ConV.wait()
                    self.get_action_ConV.release()

                    self.get_state_Conv.acquire()
                    self.get_state_Conv.wait()
                    self.state, self.reward, self.done = self.state_tuple_in
                    self.get_state_Conv.release()
                    self.state = Variable(torch.Tensor(np.asarray(self.state)).unsqueeze(0))
                    self.cum_reward += self.reward
                    self.rewards.append(self.reward)
                    if self.done:
                        self.cum_done += 1
                        self.av_reward += self.cum_reward
                        self.cum_reward = 0
                        self.episode_length = 0
                    if self.done:
                        break
                R = torch.zeros(1, 1)
                if not self.done:
                    _,_,v = self.model(self.state)
                R = v.data
                self.values.append(Variable(R))
                R = Variable(R)
                A = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(self.rewards))):
                    td = self.rewards[i] + self.params.gamma*self.values[i+1].data[0,0] - self.values[i].data[0,0]
                    A = float(td) + self.params.gamma*self.params.gae_param*A
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
                mu_old, sigma_sq_old, v_pred_old = model_old(batch_states.detach())
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
                vfloss2 = (v_pred_clipped - batch_returns)**2
                loss_value = 0.5*torch.mean(torch.max(vfloss1, vfloss2))
                # entropy
                loss_ent = -self.params.ent_coeff*torch.mean(probs*torch.log(probs+1e-5))
                # total
                total_loss = (loss_clip + loss_value + loss_ent)
                #print(total_loss.data[0])
                # before step, update old_model:
                model_old.load_state_dict(self.model.state_dict())
                # prepare for step
                total_loss.backward(retain_variables=True)
                #ensure_shared_grads(model, shared_model)
                #shared_model.cum_grads()
                self.shared_grad_buffers.add_gradient(self.model)

                self.atomicInt.inc()

                self.ChiefConV.acquire()
                self.ChiefConV.wait()
                self.ChiefConV.release()

            self.memory.clear()