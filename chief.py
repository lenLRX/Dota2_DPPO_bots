import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
import time
import datetime


def chief(params, CheifConV, atomic_counter, shared_model, shared_grad_buffers, optimizer, num_iter):
    i = num_iter
    while True:
        CheifConV.wait()
        # workers will wait after last loss computation
        if True:
            i = i + 1
            #print(shared_grad_buffers.grads['mu.weight_grad'])
            for n,p in shared_model.named_parameters():
                p._grad = Variable(shared_grad_buffers.grads[n+'_grad'])
            optimizer.step()
            shared_grad_buffers.reset()
            print('update')
            if i % 100 == 0:
                torch.save(shared_model.state_dict(),"./model/" + str(i))
