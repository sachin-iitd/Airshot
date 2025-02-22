import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj
from torch import Tensor

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, batchsize, bias=False):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batchsize = batchsize
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, input, adj):

        support = torch.einsum("jik,kp->jip",input,self.weight)
        if self.bias is not None:
            support = support + self.bias
        support = torch.reshape(support,[support.size(0),-1])
        output = torch.spmm(adj, support)
        output = torch.reshape(output,[output.size(0),self.batchsize,-1])
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# vanilla GCN
class PolicyNet(nn.Module):

    def __init__(self,args,statedim):
        super(PolicyNet, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList()
        for i in range(len(args.pnhid)):
            if (i == 0):
                self.gcn.append(GraphConvolution(statedim, args.pnhid[i], args.batchsize, bias=True))
            else:
                self.gcn.append(GraphConvolution(args.pnhid[i - 1], args.pnhid[i], args.batchsize, bias=True))
        self.output_layer = nn.Linear(args.pnhid[-1], 1, bias=False)

    def forward(self, state, adj):
        x = state.transpose(0, 1)
        for layer in self.gcn:
            x = F.relu(layer(x, adj))
        x = self.output_layer(x).squeeze(-1).transpose(0, 1)
        return x

class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int, bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(WeightedSAGEConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        in_channels = (in_channels, in_channels)
        self.lin_l = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                normalize: Tensor) -> Tensor:
        x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, norm=normalize)
        out = self.lin_l(out)
        out += self.lin_r(x[1])
        return out

    def message(self, x_j: Tensor, norm) -> Tensor:
        return x_j * norm.view(-1, 1)

class PolicyNetSageMax(torch.nn.Module):
    def __init__(self,args,statedim):
        super().__init__()
        self.conv1Max = SAGEConv(statedim, 14)
        self.conv1Max.aggr = 'max'
        self.conv2Max = SAGEConv(14, 8)
        self.conv2Max.aggr = 'max'
        self.conv4 = nn.Linear(8, 5)
        self.conv5 = nn.Linear(5, 1)
        self.reset()

    def forward(self, state, normadj):
        adj = normadj.coalesce()
        x = F.relu(self.conv1Max(state, adj.indices()))
        x = F.relu(self.conv2Max(x, adj.indices()))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x.squeeze()

    def reset(self):
        self.conv1Max.reset_parameters()
        self.conv2Max.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
