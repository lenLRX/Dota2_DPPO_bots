import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp

from utils import *

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.params = Params()
        self.h_size_1 = 20
        self.h_size_2 = 20
        h_size_1 = self.h_size_1
        h_size_2 = self.h_size_2
        self.num_inputs = num_inputs
        self.p_self_input = nn.Linear(num_inputs["self_input"],h_size_1)
        self.p_ally_creep_input = nn.Linear(num_inputs["ally_input"],h_size_1)

        self.p_cat_layer = nn.Linear(h_size_1 * 2,h_size_2)

        self.lstm = nn.LSTM(h_size_2,h_size_2)

        self.p_fc = nn.Linear(h_size_2, h_size_2)

        self.v_fc = nn.Linear(h_size_2, h_size_2)

        self.mu = nn.Linear(h_size_2, num_outputs)
        self.log_std = nn.Linear(h_size_2, num_outputs)
        self.v = nn.Linear(h_size_2,1)
        for name, p in self.named_parameters():
            # init parameters
            if 'bias' in name:
                p.data.fill_(0)
            '''
            if 'mu.weight' in name:
                p.data.normal_()
                p.data /= torch.sum(p.data**2,0).expand_as(p.data)'''
        # mode
        self.train()
    
    def init_lstm(self):
        self.lstm_hidden = (Variable(torch.zeros(1,1,self.h_size_2)),
                           Variable(torch.zeros(1,1,self.h_size_2)))

    def forward(self, inputs, hidden = None):
        if hidden != None:
            self.lstm_hidden = hidden
        # actor
        self_input = inputs["self_input"]
        self_input_out = F.tanh(self.p_self_input(self_input)).view(-1,self.h_size_1)

        ally_creep_inputs = inputs["ally_input"]
        #print(ally_creep_inputs)
        avg_creep = None

        #print("training")
        _temp = []
        for item in ally_creep_inputs:
            if isinstance(item,list):
                item = item[0]
            _out1 = F.tanh(
                        self.p_ally_creep_input(item)).view(-1,self.h_size_1)
            _temp.append(
                    torch.mean(_out1,0).view(-1,self.h_size_1))
        avg_creep = torch.cat(_temp,0)
        #print(self_input_out.size(),avg_creep.size())

        cat_out = F.tanh(
            self.p_cat_layer(
                torch.cat(
                    (self_input_out,avg_creep),1
                    )
                )
            )
        #print(cat_out)

        lstm_out,self.lstm_hidden = self.lstm(cat_out,self.lstm_hidden)
        
        x = F.tanh(self.p_fc(lstm_out))
        mu = F.tanh(self.mu(x))
        #log_std should in [0,10]
        #when error is great , log_std is great, log_std_out is great,great randomness
        log_std = self.log_std(x)
        log_std_out = torch.exp(F.relu(log_std) - self.params.log_std_bound)
        # critic
        x = F.tanh(self.v_fc(cat_out))
        v = self.v(x)
        return mu, log_std_out, v, log_std

class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)


class Shared_obs_stats():
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self._obs = {}
        for k in num_inputs:
            self._obs[k] = _s_Shared_obs_stats(num_inputs[k])

    def observes(self, obs):
        for k in self.num_inputs:
            if True:
                obj = obs[k]
                if isinstance(obj[0],list):
                    #nested
                    for l in obj:
                        self._obs[k].observes(l)
                else:
                    self._obs[k].observes(obj)

    def normalize(self, inputs):
        _out = {}
        for k in self.num_inputs:
            _out[k] = self._obs[k].normalize(inputs[k])
            if isinstance(inputs[k][0],list):
                _out[k] = [_out[k]]
        return _out



class _s_Shared_obs_stats():
    def __init__(self, num_inputs):
        pass

    def observes(self, obs):
        # observation mean var updates
        #print(obs)
        #obs =  torch.FloatTensor(obs)

        return

        obs = Variable(torch.FloatTensor(obs))
        x = obs.data.squeeze()
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        inputs = Variable(torch.FloatTensor(inputs))
        if len(inputs.size()) == 1:
            inputs = torch.unsqueeze(inputs,0)
        return inputs
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs))
        obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp((inputs-obs_mean)/obs_std, -5., 5.)
