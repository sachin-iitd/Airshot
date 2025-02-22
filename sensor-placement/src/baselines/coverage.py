import torch

def coverage_placement(batchsize,pool,val):
    action = []
    for i in range(batchsize):
        sel = [j for j in range(len(val)) if j not in pool[i]]
        v = val[pool[i]][:, sel].sum(axis=1)
        action_inpool = torch.argmax(v)
        action.append(pool[i, action_inpool])
    return action
