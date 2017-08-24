import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
import time

def chief(params, CheifConV, counter, shared_model, shared_grad_buffers, optimizer):

    while True:
        time.sleep(1)

        # workers will wait after last loss computation
        if counter.getVal() > params.update_treshold:
            #print(shared_grad_buffers.grads['mu.weight_grad'])
            for n,p in shared_model.named_parameters():
                p._grad = Variable(shared_grad_buffers.grads[n+'_grad'])
            optimizer.step()
            counter.setVal(0)
            shared_grad_buffers.reset()
            CheifConV.acquire()
            CheifConV.notify_all()
            CheifConV.release()
            print('update')
