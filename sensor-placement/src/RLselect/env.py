import torch
import torch.nn.functional as F
import numpy as np
from src.RLselect.player import Player
from src.RLselect.utils import *

def logprob2Prob(logprobs):
    probs = F.softmax(logprobs, dim=2)
    return probs

def normalizeEntropy(entro,classnum): #this is needed because different number of classes will have different entropy
    maxentro = np.log(float(classnum))
    entro = entro/maxentro  
    return entro

def prob2Logprob(probs):
    logprobs = torch.log(probs)
    return logprobs

def perc(input):
    # the biger valueis the biger result is
    numnode = input.size(-2)
    return torch.argsort(torch.argsort(input, dim=-2), dim=-2) / float(numnode)

def degprocess(deg):
    return torch.clamp_max(deg / 20., 1.)

def localdiversity_pmval(pmvals,adj,deg,device):
    indices = adj.coalesce().indices()
    N = adj.size()[0]
    edgevals = pmvals[:,indices.transpose(0,1)]
    headvals = edgevals[:,:,0]
    tailvals = edgevals[:,:,1]

    pmdiff = abs(headvals - tailvals).transpose(0,1)
    sparse_output_pmdiff = torch.sparse.FloatTensor(indices,pmdiff,size=torch.Size([N,N,pmdiff.size(-1)]))
    sum_pmdiff = torch.sparse.sum(sparse_output_pmdiff,dim=1).to_dense().transpose(0,1)
    mean_pmdiff = sum_pmdiff/deg #+1e-30)
    # normalize
    mean_pmdiff_max = mean_pmdiff.max(dim=1, keepdim=True).values
    mean_pmdiff /= mean_pmdiff_max+1e-10
    return mean_pmdiff

def localdiversity(probs,adj,deg):
    indices = adj.coalesce().indices()
    N =adj.size()[0]
    classnum = probs.size()[-1]
    #maxentro = np.log(float(classnum))
    edgeprobs = probs[:,indices.transpose(0,1),:]
    headprobs = edgeprobs[:,:,0,:]
    tailprobs = edgeprobs[:,:,1,:]
    kl_ht = (torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*tailprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*tailprobs,dim=-1)).transpose(0,1)
    kl_th = (torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*headprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*headprobs,dim=-1)).transpose(0,1)
    sparse_output_kl_ht = torch.sparse.FloatTensor(indices,kl_ht,size=torch.Size([N,N,kl_ht.size(-1)]))
    sparse_output_kl_th = torch.sparse.FloatTensor(indices,kl_th,size=torch.Size([N,N,kl_th.size(-1)]))
    sum_kl_ht = torch.sparse.sum(sparse_output_kl_ht,dim=1).to_dense().transpose(0,1)
    sum_kl_th = torch.sparse.sum(sparse_output_kl_th,dim=1).to_dense().transpose(0,1)
    mean_kl_ht = sum_kl_ht/(deg[0]+1e-30)
    mean_kl_th = sum_kl_th/(deg[0]+1e-30)
    # normalize
    mean_kl_ht_max = mean_kl_ht.max(dim=1, keepdim=True).values
    mean_kl_th_max = mean_kl_th.max(dim=1, keepdim=True).values
    if not torch.sum(mean_kl_ht_max < 1e-30).item():
        mean_kl_ht = mean_kl_ht / mean_kl_ht_max
    if not torch.sum(mean_kl_th_max < 1e-30).item():
        mean_kl_th = mean_kl_th / mean_kl_th_max
    return mean_kl_ht,mean_kl_th


class Env(object):
    ## an environment for multiple players testing the policy at the same time
    def __init__(self,players,args,device='cpu'):
        '''
        players: a list containing main player (many task) (or only one task
        '''
        self.players = players
        self.args = args
        self.device = device
        self.nplayer = len(self.players)
        self.graphs = [p.G for p in self.players]
        featdim =-1
        self.statedim = self.getState(0).size(featdim)


    def step(self,actions=None,playerid=0,rew=True):
        p = self.players[playerid]
        if actions is not None:
            p.query(actions)
        loss = p.trainOnce()
        if not rew:
            return None
        rewardpool = p.validation(test=-1, rerun=True)
        reward = p.validation(test=False, rerun=True)
        rewardtest = p.validation(test=True, rerun=True)
        return reward, rewardtest, loss, rewardpool


    def getState(self,playerid=0):
        p = self.players[playerid]
        output = p.allnodes_output[2] if not self.args.classify else logprob2Prob(p.allnodes_output[2])
        state = self.makeState(output,p.trainmaskGP,playerid)
        return state


    def reset(self,playerid=0,resplit=False):
        self.players[playerid].reset(fix_test=False,resplit=resplit)

    
    def makeState(self, probs, selected,playerid):
        # probs contain RMSE for regression case
        G = self.players[playerid].G

        if G.GP:
            # Average RMSE values for the same Lat-Long (over different time)
            rmse = None
            for i,nodes in enumerate(G.GPmap[0] if G.undirected is None else G.GPmap):
                v = torch.mean(probs[:,nodes],axis=1)
                v = v.reshape((probs.shape[0],-1))
                if rmse is None:
                    rmse = v
                else:
                    rmse = torch.cat((rmse, v), axis=1)
            probs = rmse

        features = []
        if self.args.use_entropy:
            entro = entropy(probs)
            entro = normalizeEntropy(entro, probs.size(-1))  ## in order to transfer
            features.append(entro)
        if self.args.use_rmse:
            features.append(probs/probs.max())
        if self.args.use_local_diversity:
            if self.args.classify:
                mean_kl_ht,mean_kl_th = localdiversity(probs,G.normadjGP,G.deg)
                features.extend([mean_kl_ht, mean_kl_th])
            else:
                diver = localdiversity_pmval(probs,G.normadjGP,G.deg[0],self.device)
                features.append(diver)
        if self.args.use_select:
            features.append(selected)
        if self.args.use_degree:
            features.append(G.deg)
        if self.args.use_pagerank:
            features.append(G.pagerank)

        state = torch.stack(features, dim=-1)

        #if torch.sum(torch.isnan(state)):
        #    print("State with NaN")

        return state