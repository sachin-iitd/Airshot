import torch

def pagerank_placement(batchsize,pool,val):
    action = []
    for i in range(batchsize):
        v = val[i, pool[i]]
        action_inpool = torch.argmax(v)
        action.append(pool[i, action_inpool])
    return action
