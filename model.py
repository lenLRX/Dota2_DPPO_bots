import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn.init as init

from utils import *


class Model(nn.Module):
    '''
           noop
    input->move->9 directions
           attack->choose 1 target
    '''
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.params = Params()
        self.h_size_1 = 20
        self.h_size_2 = 20
        self.spatial_res = self.params.num_outputs
        h_size_1 = self.h_size_1
        h_size_2 = self.h_size_2
        self.env_input_dim = num_inputs["env_input"]
        self.atk_target_dim = num_inputs["atk_target"]

        #input layer
        self.input_layer = nn.Linear(self.env_input_dim,h_size_2)
        self.init_layer(self.input_layer)

        #decision layers
        self.decision_layer = nn.Linear(h_size_2,3)
        self.init_layer(self.decision_layer)

        #value layers
        self.v_input_1 = nn.Linear(self.env_input_dim,h_size_2)
        self.v_input_2 = nn.Linear(h_size_2,h_size_2)

        self.init_layer(self.v_input_1)
        self.init_layer(self.v_input_2)

        self.v = nn.Linear(h_size_2,1)
        self.init_layer(self.v)

        #move layers
        self.input_move_layer = nn.Linear(self.env_input_dim,h_size_2)
        self.init_layer(self.input_move_layer)
        self.spatial = nn.Linear(h_size_2, self.spatial_res ** 2)
        self.init_layer(self.spatial)

        #attack
        self.attack_layer = nn.Linear(param.num_inputs["atk_target"], 1)

        # mode
        self.train()
    
    def init_layer(self,layer):
        init.xavier_uniform(layer.weight, gain=np.sqrt(2))
        init.constant(layer.bias, 0.01)

    def init_lstm(self):
        self.lstm_hidden = (Variable(torch.zeros(1,1,self.h_size_2)),
                           Variable(torch.zeros(1,1,self.h_size_2)))

    def forward(self,inputs):
        atk_target = None
        atk_target_log = None
        move_target = None
        move_target_log = None

        _env_input = Variable(torch.FloatTensor(inputs["env_input"])).view(1,1,-1)

        v = _env_input
        v = F.sigmoid(self.v_input_1(v)).view(-1,self.h_size_2)
        v = F.sigmoid(self.v_input_2(v)).view(-1,self.h_size_2)
        v_out = self.v(v)

        p = _env_input
        input_layer_out = F.sigmoid(self.input_layer(p)).view(-1,self.h_size_2)
        decision_layer_out = self.decision_layer(input_layer_out).view(-1,param.num_outputs)
        decision_layer_softmax_out = F.softmax(decision_layer_out)
        decision_layer_log_out = F.log_softmax(decision_layer_out)

        decision = np.argmax(decision_layer_softmax_out.data.numpy()[0])

        if 0 == decision:
            #no op
            pass
        elif 1 == decision:
            #move
            move_layer_out = F.sigmoid(self.input_move_layer(_env_input)).view(-1,self.h_size_2)
            spatial_layer_out = self.spatial(move_layer_out).view(-1,self.spatial_res**2)
            spatial_layer_softmax_out = F.softmax(spatial_layer_out)
            spatial_layer_log_out = F.log_softmax(spatial_layer_out)
            move_target = np.argmax(spatial_layer_softmax_out.data.numpy()[0])
            move_target_log = spatial_layer_log_out
            assert(spatial_layer_log_out.size()[1] == 9)
        elif 2 == decision:
            #attack
            if len(inputs["target_input"]) > 0:
                _raw_inputs = Variable(torch.FloatTensor(inputs["target_input"]))
                targets = []
            
                for t in _raw_inputs:
                    targets.append(self.attack_layer(t))
                _ts = torch.cat(targets)
                
                _ts_softmax = F.softmax(_ts)
                _ts_log = F.log_softmax(_ts)
                atk_target = np.argmax(_ts_softmax.data.numpy()[0])
                atk_target_log = _ts_log
        else:
            raise Exception("unknown decision")


        return v_out, decision, decision_layer_log_out,\
                move_target, move_target_log,\
                atk_target, atk_target_log

class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.zeros(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)
