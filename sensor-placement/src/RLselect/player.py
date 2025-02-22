# individual player who takes the action and evaluates the effect
import torch
import torch.nn as nn
import time
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import scipy
from src.RLselect.common import *
from src.RLselect.utils import *
#from src.RLselect.classificationnet import GCN
from src.RLselect.classificationnet import *

class Player(nn.Module):

    def __init__(self,G,args,device=None,rank=0):

        super(Player,self).__init__()
        self.G = G
        self.args = args
        self.device = device
        self.rank = rank
        self.batchsize = args.batchsize
        self.sageGCN = args.sageGCN
        self.lossfn = args.lossfn
        self.accmask = torch.tensor([1.0, 0.8, 0.5, 0]).to(self.device)  # Last should be zero
        self.analyze = [0] * 32
        self.analyze[0] = -2

        self.losslabel = None
        self.fulllabel = None
        self.retain_graph = False

        switcher = {'gcn':GCN, 'gcnn':GCNN, 'rf':RF}
        net = switcher[self.args.predictnet]
        print('Predictor', args.predictnet, net.__name__) #, ', HID:', *args.nhid)
        if self.lossfn == 0:
            self.net = net(self.G.stat['nfeat'],args.nhid,self.G.stat['nclass'],args.batchsize,args.dropout,True,device=self.device).to(self.device)
            self.loss_func = F.nll_loss
            self.fulllabel = self.G.Y.expand([self.batchsize] + list(self.G.Y.size())).to(self.device)
        elif self.lossfn == 1:
            self.net = net(self.G.stat['nfeat'],args.nhid,self.G.stat['nclass'],args.batchsize,args.dropout,softmax=True,regres=2,device=self.device).to(self.device)
            self.loss_func = F.mse_loss
            self.loss_func2 = F.nll_loss
            self.retain_graph = True
            self.fulllabel = self.G.Y.expand([self.batchsize] + list(self.G.Y.size())).type(torch.LongTensor).to(self.device)
            self.losslabel = self.G.YV.expand([self.batchsize] + list(self.G.YV.size())).type(torch.FloatTensor).to(self.device)
        elif self.lossfn == 2:
            self.net = net(self.G.stat['nfeat'],args.nhid,self.G.stat['nclass'],args.batchsize,args.dropout,softmax=False,regres=2,device=self.device).to(self.device)
            self.loss_func = F.mse_loss
            self.losslabel = self.G.YV.expand([self.batchsize] + list(self.G.YV.size())).type(torch.FloatTensor).to(self.device)
        else:
            print('Undefined LossFn')
            exit(0)
        if self.losslabel is None:
            self.losslabel = self.fulllabel.type(torch.LongTensor).to(self.device)  # typecast for NLL

        self.reset(fix_test=False)
        self.count = 0
        self.setValTestMask(self.G)

    # def net_wrap(self, X, normadj):
    def net_wrap(self, pool, X, Y):

        # all_nodes <- out2
        # Loss <- out
        outLoss, outAcc, outState = None,  None, None
        outLoss = self.net(pool[:,:-1], X, Y)     # LogSoftmaxBackward Shape:(Batch,Class,Samples)

        if self.lossfn == 2:
            outAcc = outLoss
            if Y is None:
                outState = outLoss
            else:
                outState = torch.sqrt((self.losslabel - (outLoss.detach() if self.G.undirected is not None else outLoss)) ** 2)
        elif self.lossfn == 0:
            outAcc = outLoss
        elif self.lossfn == 1:
            outLoss, outAcc = outLoss

        if outState is None:
            outState = outAcc.transpose(1,2).detach()

        return outLoss, outAcc if self.G.undirected is None else outAcc.detach(), outState

    def setValTestMask(self, G):
        self.mask = G

    def lossWeighting(self,epoch):
        return min(epoch,10.)/10.

    def query(self,nodes):
        #self.trainmask[[x for x in range(self.batchsize)],nodes] = 1.
        for i,n in enumerate(nodes):
            self.trainmaskGP[i, n] = 1.
            self.trainmask[i, self.G.GPmap[0][n]] = 1.

    def numTrainingNodes(self):
        if self.G.GP:
            mask = self.mask.maskGP[0]
        else:
            mask = self.mask.testmask[0] + self.mask.valmask[0]
        return len(mask) - int(sum(mask).item())

    def getPool(self,reduce=True):
        mask = self.trainmaskGP # + self.mask.maskGP #+self.testmask+self.valmask
        row,col = torch.where(mask<0.1)
        if reduce:
            row, col = row.cpu().numpy(),col.cpu().numpy()
            pool = []
            for i in range(self.batchsize):
                pool.append(col[row==i])
            return pool
        else:
            return row,col

    def trainOnce(self,log=False):
        if not self.sageGCN:
            trainmask = self.trainmask
            nlabeled = torch.sum(trainmask) / self.batchsize
        else:
            raise 'Validate this'
            trainmask = self.trainmask[0]
            nlabeled = torch.sum(trainmask)
        self.net.train()
        if self.G.undirected is None:
            XY = None
            for i in range(self.batchsize):
                xy = self.G.pool[trainmask[i].type(torch.BoolTensor), :][np.newaxis,...].cpu()
                if XY is None:
                    XY = xy
                elif XY.shape[1] == xy.shape[1]:
                    XY = np.vstack((XY, xy))
                else:
                    nc = xy.shape[-1]
                    diff = abs(XY.shape[1] - xy.shape[1])
                    x0 = np.array([0] * (nc * diff)).reshape(1, diff, nc)
                    if XY.shape[1] < xy.shape[1]:
                        XY = np.hstack((XY, x0))
                    else:
                        xy = np.hstack((xy, x0))
                    XY = np.vstack((XY, xy))

            X = torch.from_numpy(XY[:,:,:-1]).to(self.device)
            Y = torch.from_numpy(XY[:,:,-1]).to(self.device)
            self.allnodes_output = self.net_wrap(self.G.pool,X,Y)
            losses = self.allnodes_output[-1]
            if self.args.predictforecast:
                losses = losses[:,self.G.pool_flag]
            l = torch.sum(losses,dim=1)/nlabeled # *self.lossWeighting(float(nlabeled.cpu()))
            l = l.detach().cpu().numpy().tolist()
            return l
        else:
            self.opt.zero_grad()
            out = self.net_wrap(self.G.X,self.G.normadj)
            losses = self.loss_func(out[0],self.losslabel,reduction="none")
            # TODO optimize
            loss = torch.sum(losses*trainmask)/nlabeled*self.lossWeighting(float(nlabeled.cpu()))
            loss.backward(retain_graph=self.retain_graph)
            l = torch.sum(losses*trainmask,dim=1)/nlabeled*self.lossWeighting(float(nlabeled.cpu()))
            l = l.detach().cpu().numpy().tolist()
            if self.lossfn == 1:
                losses2 = self.loss_func2(out[2], self.fulllabel, reduction="none")
                loss = torch.sum(losses2 * trainmask) / nlabeled * self.lossWeighting(float(nlabeled.cpu()))
                loss.backward()

            self.loss_acc += sum(l)
            self.loss_ctr += 1
            self.opt.step()
            self.allnodes_output = out
            return l


    def validation1(self,test=0,rerun=True):
        if rerun:
            self.net.eval()
            self.allnodes_output = self.net_wrap(self.G.X,self.G.normadj)

        out = self.allnodes_output
        out1 = (out[1] / self.G.range).to(torch.long)
        acc = []
        rng = self.mask.valtestrng[test]
        for typ in range(rng[0],rng[1]):
            index = self.mask.valtestid[typ]
            labels = self.mask.valtestlabel[typ]
            cpulabels = self.mask.valtestlabelcpu[typ]
            pmval = self.mask.valtestpmval[typ]
            pred_vals = out1[:,index]
            pred_vals2 = out[0].detach()[:,index]

            def ordinal_accuracy(labels, pred_vals):
                diff = torch.abs(pred_vals - labels)
                diff = torch.clamp(diff,max=len(self.accmask)-1)
                vals = self.accmask[diff]
                return torch.mean(vals,axis=1)

            # Reference: Scipy
            def torch_rmse(y_true, y_pred):
                mse = torch.mean((y_true - y_pred) ** 2, axis=1)
                return torch.sqrt(mse)

            ord = ordinal_accuracy(labels, pred_vals)
            rmse = torch_rmse(pmval, pred_vals2)
            #f1s = f1_loss(labels, pred_vals)
            if self.lossfn == 2:
                for i in range(self.batchsize):
                    f1s = f1_score(cpulabels, pred_vals[i].cpu(), average="micro")
                    acc.append([f1s, ord[i].item(), rmse[i].item()])
            else:
                raise 'Not supported'

        # logger.info("validation acc {}".format(acc))
        return list(zip(*acc))

    # Reference: Scipy
    def torch_rmse(self, y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)  # , axis=1)
        return torch.sqrt(mse)

    def validation(self,test=0,rerun=True):
        acc = []
        if test < 0:
            for i in range(self.batchsize):
                XY = self.G.pool
                if test == -2:
                    raise 'Update this'
                    XY = XY[(1-self.trainmask[i]).type(torch.BoolTensor), :]
                pred_vals, l1, l2 = self.net_wrap(XY.cpu(), None, None)
                pmval = XY[:,-1]
                if self.args.predictforecast:
                    pred_vals = pred_vals[:, self.G.pool_flag]
                    pmval = pmval[self.G.pool_flag]
                acc.append([0, 0, self.torch_rmse(pmval, pred_vals[i]).item()])
        else:
            XY = self.G.test if test>0 else self.G.valid
            output = self.allnodes_output
            if rerun or self.args.undirected is None:
                # self.net.eval()
                output = self.net_wrap(XY,None,None)

            pmval = XY[:,-1]
            pred_vals = output[0]
            self.pred_vals = pred_vals.clone()
            if self.args.predictforecast:
                F = self.G.test_flag if test > 0 else self.G.valid_flag
                pred_vals = pred_vals[:, F]
                pmval = pmval[F]

            for i in range(self.batchsize):
                acc.append([0, 0, self.torch_rmse(pmval, pred_vals[i]).item()])

            if test > 0 and self.G.testidx is not None:
                i = 0
                for j in self.G.testidx:
                    for b in range(self.batchsize):
                        acc.append([0,0,self.torch_rmse(pmval[i:i+j], pred_vals[b][i:i+j]).item()])
                    i += j

        return list(zip(*acc))

    def trainRemain(self):
        for i in range(self.args.remain_epoch):
            l = self.trainOnce()
        return l, 0 # self.loss_acc/self.loss_ctr


    def reset(self,resplit=False,fix_test=True):
        if resplit:
            self.G.makeValTestMask(self.args)
            self.setValTestMask(self.G)
        self.trainmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).to(self.device)
        self.trainmaskGP = torch.zeros((self.batchsize,self.G.stat['nnodeGP'])).to(torch.float).to(self.device)
        for node in range(len(self.G.loc_init)):
            self.query(torch.tensor([node for _ in range(self.batchsize)]))
        self.net.reset()
        self.opt = torch.optim.Adam(self.net.parameters(),lr=self.args.lr,weight_decay=5e-4) if self.G.undirected is not None else None
        # First pass only 1 sample per batch
        self.trainOnce()
        XY = self.G.pool[:self.batchsize].unsqueeze(1)
        self.allnodes_output = self.net_wrap(self.G.pool, XY[:,:,:-1], XY[:,:,-1])
        self.loss_acc = 0.0
        self.loss_ctr = 0
