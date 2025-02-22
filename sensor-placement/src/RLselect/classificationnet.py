import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone as skclone
import numpy as np

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batchsize, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batchsize = batchsize
        self.weight = nn.Parameter(torch.FloatTensor(batchsize,in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(batchsize*out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.einsum("jik,ikp->jip",input,self.weight)
        support = torch.reshape(support,[support.size(0),-1])
        # support = torch.mm(input, self.weight)
        if self.bias is not None:
            support = support + self.bias
        output = torch.spmm(adj, support)
        output = torch.reshape(output,[output.size(0),self.batchsize,-1])
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, batchsize, dropout=0.5, softmax=True, regres=0, bias=False, device='cuda'):
        super(GCN, self).__init__()
        self.device = device
        self.dropout_rate = dropout
        self.softmax = softmax
        self.batchsize = batchsize
        self.regres = regres
        self.gc1 = GraphConvolution(nfeat, nhid[0], batchsize, bias=bias)
        self.gc2 = GraphConvolution(nhid[0], nclass, batchsize, bias=bias)
        self.dropout = nn.Dropout(p=dropout,inplace=False)
        if self.regres==1:
            self.reg = nn.Linear(nclass*batchsize, 1*batchsize)
        elif self.regres==2:
            self.reg = nn.ModuleList([nn.Linear(nclass, 1).to(device) for _ in range(batchsize)])

    def forward(self, x, adj):
        x = x.expand([self.batchsize]+list(x.size())).transpose(0,1)
        x = self.dropout(x)
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)

        if self.regres == 2:
            x = x.transpose(0, 1)
            y = torch.zeros((x.shape[1],0)).to(self.device)
            for reg,xx in zip(self.reg,x):
                y = torch.cat((y,reg(xx)), dim=1)
            x = y.T
        elif self.regres == 1:
            x = x.reshape((x.shape[0], -1))
            x = self.reg(x).T
        elif self.softmax:
            y = x.transpose(0, 1).transpose(1, 2)
            x = F.log_softmax(y, dim=1)
        else:
            y = x.transpose(0, 1).transpose(1, 2)
            x = y.squeeze()
        return x

    def reset(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        if self.regres==2:
            for reg in self.reg:
                reg.reset_parameters()
        elif self.regres==1:
            self.reg.reset_parameters()


class GCNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, batchsize, dropout=0.5, softmax=True, regres=0, bias=False, device='cuda'):
        super().__init__()
        self.device = device
        self.dropout_rate = dropout
        self.softmax = softmax
        self.batchsize = batchsize

        dims = [nfeat] + nhid + [nclass]
        self.gc = nn.ModuleList([GraphConvolution(dims[i], dims[i+1], batchsize, bias=bias) for i in range(len(nhid)+1)])
        self.reg = nn.ModuleList([nn.Linear(nclass, 1).to(device) for _ in range(batchsize)])
        self.dropout = nn.Dropout(p=dropout,inplace=False)

    def forward(self, x, adj):
        x = x.expand([self.batchsize]+list(x.size())).transpose(0,1)
        for gc in self.gc[:-1]:
            x = self.dropout(x)
            x = F.relu(gc(x, adj))
        x = self.dropout(x)
        x = self.gc[-1](x, adj)

        x = x.transpose(0, 1)
        y = torch.zeros((x.shape[1],0)).to(self.device)
        for reg,xx in zip(self.reg,x):
            y = torch.cat((y,reg(xx)), dim=1)
        return y.T

    def reset(self):
        for gc in self.gc:
            gc.reset_parameters()
        for reg in self.reg:
            reg.reset_parameters()


class RF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, batchsize, dropout=0.5, softmax=True, regres=0, bias=False, device='cuda'):
        super().__init__()
        self.batchsize = batchsize
        self.device = device
        self.reset()

    def forward(self, pool, input, output):
        if output is not None:
            for i in range(self.batchsize):
                self.model[i] = skclone(self.model[i])
                self.model[i].fit(input[i].cpu(), output[i].cpu() if len(output.shape)>=2 else output.cpu())
        pred = []
        for i in range(self.batchsize):
            pred.append(self.model[i].predict(pool.cpu()))
        return torch.from_numpy(np.array(pred)).to(self.device)

    def reset(self):
        self.model = [RandomForestRegressor(n_estimators=100, random_state=0) for _ in range(self.batchsize)]
