import torch

def random_placement(batchsz,pool):
    action_inpool = torch.randint(pool.shape[1], (batchsz,))
    action = pool[[x for x in range(batchsz)], action_inpool]
    return action
