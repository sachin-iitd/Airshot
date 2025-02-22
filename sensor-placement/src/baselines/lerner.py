import torch

def lerner_placement(batchsize,pool,val):
    action = []
    for i in range(batchsize):
        v = val[pool[i]]
        action_inpool = torch.argmax(v)
        action.append(pool[i, action_inpool])
    return action
