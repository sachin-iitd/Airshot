import numpy as np
import scipy.sparse as sp
import warnings
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
import torch
import math

from src.RLselect.common import *


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features

def column_normalize(tens):
    ret = tens - tens.mean(axis=0)
    return ret

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_add_diag=adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj_add_diag)
    return adj_normalized.astype(np.float32) #sp.coo_matrix(adj_unnorm)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

##=========================================================================

mask = [1.0, 0.8, 0.5]
def ordinal_accuracy(labels, y_pred):
    diff = np.abs(labels - y_pred)
    vals = [mask[d] if d < len(mask) else 0 for d in diff]
    return np.sum(vals)/len(labels)

def accuracy(y_pred, labels, num=3):
    if len(labels.size())==1:
        if len(y_pred.size())>1:
            y_pred = y_pred.max(1)[1].type_as(labels)
        else:
            y_pred = y_pred.type_as(labels)
        y_pred=y_pred.detach().cpu().numpy()
        labels=labels.cpu().numpy()


    elif len(labels.size())==2:
        y_pred=(y_pred > 0.).cpu().detach().numpy()
        labels=labels.cpu().numpy()

    rmse = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mic = f1_score(labels, y_pred, average="micro")
        mac = ordinal_accuracy(labels, y_pred)
        if num >= 3:
            rmse = mean_squared_error(labels, y_pred, squared=False)

    return mic,mac,rmse

def mean_std(L):
    if type(L)==np.ndarray:
        L=L.tolist()
    m=sum(L)/float(len(L))
    bias=[(x-m)**2 for x in L]
    std=math.sqrt(sum(bias)/float(len(L)-1))
    return float(m),float(std)

##==========================================================================

def entropy(tens):
    assert type(tens)==torch.Tensor and len(tens.size())==3,"calculating entropy of wrong size"
    entropy = - torch.log(torch.clamp(tens, min=1e-7)) * tens
    entropy = torch.sum(entropy, dim=2)
    return entropy


##==========================================================================


class AverageMeter(object):
    def __init__(self,name='',ave_step=10):
        self.name = name
        self.ave_step = ave_step
        self.history =[]
        self.history_extrem = None
        self.S=5

    def update(self,data):
        if data is not None:
            self.history.append(data)
            self.avg = None

    def __call__(self):
        if len(self.history) == 0:
            self.avg = None
        elif self.avg is None:
            cal = self.history[-self.ave_step:]
            self.avg = sum(cal)/float(len(cal))
        return self.avg

    def should_save(self):
        if len(self.history)>self.S*2 and sum(self.history[-self.S:])/float(self.S)> sum(self.history[-self.S*2:])/float(self.S*2):
            if self.history_extrem is None :
                self.history_extrem =sum(self.history[-self.S:])/float(self.S)
                return False
            else:
                if self.history_extrem < sum(self.history[-self.S:])/float(self.S):
                    self.history_extrem = sum(self.history[-self.S:])/float(self.S)
                    return True
                else:
                    return False
        else:
            return False


class AverageMeterSimple(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []

    def update(self,data):
        self.history.append(data)

    def __call__(self):
        return sum(self.history)/float(len(self.history))


#===========================================================

def inspect_grad(model):
    name_grad = [(x[0], x[1].grad) for x in model.named_parameters() if x[1].grad is not None]
    name, grad = zip(*name_grad)
    assert not len(grad) == 0, "no layer requires grad"
    mean_grad = [torch.mean(x) for x in grad]
    max_grad = [torch.max(x) for x in grad]
    min_grad = [torch.min(x) for x in grad]
    getLogger().info("name {}, mean_max min {}".format(name,list(zip(mean_grad, max_grad, min_grad))))

def inspect_weight(model):
    name_weight = [x[1] for x in model.named_parameters() if x[1].grad is not None]
    print("network_weight:{}".format(name_weight))


#==============================================================

def common_rate(counts,prediction,seq):
    summation = counts.sum(dim=1, keepdim=True)
    squaresum = (counts ** 2).sum(dim=1, keepdim=True)
    ret = (summation ** 2 - squaresum) / (summation * (summation - 1)+1)
    equal_rate=counts[seq,prediction].reshape(-1,1)/(summation+1)
    return ret,equal_rate

