import os
import networkx as nx
import pickle as pkl
import pandas as pd
import torch
import numpy as np
from collections import OrderedDict
import time
from src.RLselect.common import *
from src.RLselect.utils import *
from sklearn import preprocessing
import datetime

class GraphLoader(object):

    def __init__(self,name, idx=0, root = "./data",undirected=True, hasX=True,hasY=True,header=True,sparse=True,args=None,nClass=10,folder=None,device=None):

        CreateLogger(args.model_dir+"/")

        self.classinY = False
        self.hasMap = True

        self.device = device
        self.nClass = nClass
        self.name = name
        self.undirected = undirected
        self.hasX = hasX
        self.hasY = hasY
        self.header = header
        self.sparse = sparse
        self.idx = idx
        self.dirname = os.path.join(root,folder) if folder else root
        self.args = args

        self.prefix = os.path.join(self.dirname,name.format(args.seasons[idx]))
        self.prefixtest = os.path.join(self.dirname,name) if args.cmntest else self.prefix

        if self.undirected is None:
            self._loaddata()
        else:
            self._load()
        self._registerStat()
        self.printStat()


    def _loadConfig(self):
        file_name = os.path.join(self.dirname,"bestconfig.txt")
        if os.path.exists(file_name):
            f = open(file_name,'r')
            L = f.readlines()
            L = [x.strip().split() for x in L]
            self.bestconfig = {x[0]:x[1] for x in L if len(x)!=0}
        else:
            self.bestconfig = {'feature_normalize': '0'}


    def _loadGraph(self, header = True, weight=0, seppolicy=False):
        file_name = self.prefix + ('.gpa' if seppolicy else '') + ".edgelist"
        if not header:
            getLogger().warning("You are reading an edgelist with no explicit number of nodes")
        if self.undirected is None:
            return None
        if self.undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()
        with open(file_name) as f:
            L = f.readlines()

        if header:
            num_node = int(L[0].strip())
            G.add_nodes_from([x for x in range(num_node)])
            L = L[1:]
        if weight and len(L[0].strip().split()) == 3:
            print('Loading edges with weights', file_name.split('/')[-1])
            edge_list = []
            for e in L:
                ee = e.strip().split()
                edge_list.append((int(ee[0]), int(ee[1]), {'weight': float(ee[2])}))
            G.add_edges_from(edge_list)
        else:
            print('Loading edges without weights')
            edge_list = [[int(x) for x in e.strip().split()[:2]] for e in L]
            nodeset = set([x for e in edge_list for x in e])
            if not header:
                G.add_nodes_from([x for x in range(max(nodeset) + 1)])
            G.add_edges_from(edge_list)
        return G

    def _loadX(self):
        self.X = pkl.load(open(self.prefix + (".x" if self.args.onehot else ".xv") +".pkl", 'rb'))
        self.X = self.X.astype(np.float32)

    def _loadY(self):
        YV = pkl.load(open(self.prefix+".y.pkl",'rb')).astype(np.float64)
        if self.classinY:
            self.Y = YV.astype(int)
        else:
            self.range = (YV.max() + 1) / self.nClass
            self.Y = (YV / self.range).astype(int)
            assert (self.Y.max() + 1) == self.nClass, 'Error in class formation {0} {1}'.format((self.Y.max() + 1), self.nClass)
        if self.args.lossfn:
            self.YV = YV

    def _loadMap(self):
        if self.GP:
            GPdata = pkl.load(open(self.prefix + ".gpa.map.pkl", 'rb'))
            if len(GPdata) == 2:
                self.GPmap, self.GPidx = GPdata
                self.GPmap2, self.GPskip = None, None
            else:
                self.GPmap, _, self.GPmap2, self.GPskip = GPdata
            GPloc = []
            for i,nodes in enumerate(self.GPmap):
                GPloc.append(nodes[0])
                self.GPmap[i] = nodes[1:]
            self.GPloc = torch.tensor(GPloc).to(self.device)

    def _getAdj(self):
        self.adj = nx.adjacency_matrix(self.G).astype(np.float32)
        if self.GP:
            self.adjGP = nx.adjacency_matrix(self.GP).astype(np.float32)

    def _toTensor(self):
        if self.sparse:
            if self.normadj:
                self.normadj = sparse_mx_to_torch_sparse_tensor(self.normadj).to(self.device)
            if self.GP:
                self.normadjGP = sparse_mx_to_torch_sparse_tensor(self.normadjGP).type(torch.DoubleTensor).to(self.device)
                adj = sparse_mx_to_torch_sparse_tensor(self.adjGP)
                deg = torch.sparse.sum(adj.to(self.device), dim=1).to_dense()
            else:
                self.normadjGP = self.normadj
                adj = sparse_mx_to_torch_sparse_tensor(self.adj)
                deg = torch.sparse.sum(adj.to(self.device), dim=1).to_dense()
        else:
            adj = torch.from_numpy(self.adj)
            self.normadj = torch.from_numpy(self.normadj).to(self.device)
            deg = torch.sum(adj.to(self.device), dim=1)
        if self.args.use_degree:
            deg /= deg.max()
            self.deg = deg.expand([self.args.batchsize]+list(deg.size()))
        else:
            self.deg = [deg]
        if self.args.use_pagerank:
            pagerank = np.array(list(nx.pagerank(self.GP if self.GP else self.G).values()))
            pagerank /= pagerank.max()
            pagerank = torch.from_numpy(pagerank).type(torch.FloatTensor).to(self.device)
            self.pagerank = pagerank.expand([self.args.batchsize] + list(pagerank.size()))
        if self.args.baseline == 'bwcentrality':
            centrality = nx.betweenness_centrality(self.GP if self.GP else self.G, None)
            centrality = torch.from_numpy(np.array(list(centrality.values())))
            self.bwcentrality = centrality.type(torch.FloatTensor).to(self.device)
        if self.X is not None and type(self.X) is not torch.Tensor:
            self.X = torch.from_numpy(self.X).to(self.device)
            self.Y = torch.from_numpy(self.Y).to(self.device)
            if self.args.lossfn:
                self.YV = torch.from_numpy(self.YV).to(self.device)

    def _load(self):
        self.G = self._loadGraph(header=self.header, weight=self.args.weight)
        self.GP = None
        if self.args.seppolicygraph:
            self.GP = self._loadGraph(header=self.header, weight=self.args.weight, seppolicy=True)
        self._loadConfig()
        if self.hasX:
            self._loadX()
        if self.hasY:
            self._loadY()
        if self.hasMap:
            self._loadMap()
        self._getAdj()

    def _make2Dgraph(self, d):

        # Select any of the below based on Scikit versions
        # from sklearn.neighbors import DistanceMetric
        from sklearn.metrics import DistanceMetric

        haversine = DistanceMetric.get_metric('haversine')
        dists = haversine.pairwise(d)
        # dists *= 6371
        w = 1 / (1 + dists)

        G = nx.Graph()
        G.add_nodes_from([x for x in range(len(d))])
        edge_list = []
        for n1 in range(len(d)):
            for n2 in range(n1+1,len(d)):
                edge_list.append((n1, n2, {'weight': float(w[n1,n2])}))
        G.add_edges_from(edge_list)
        return G, dists

    def _loaddata(self):
        self.G = None
        self.normadj = None
        self.X = None

        self._loadConfig()

        def read_csv(file):
            if 'kk' not in file:
                df = pd.read_csv(file)
                cols = ['lat', 'lon', 'time']
            else:
                df = pd.read_csv(file, parse_dates=['time'])
                cols = ['lat', 'lon', 'hour']
                if self.args.use_hour:
                    df['hour'] = [d.days * 24 + d.seconds//3600 for d in (df.time - datetime.datetime(2023,3,27))]
                elif self.args.use_day:
                    cols = ['lat', 'lon', 'day']
                    df['day'] = [(d.year - 2023) * 365 + d.day_of_year for d in df.time]
                elif self.args.baseline.startswith('mi'):
                    cols = ['lat', 'lon', 'hour']
                    df['hour'] = [d.days * 24 + d.seconds//3600 for d in (df.time - df.time.min())]
                if self.args.use_month == 1:
                    cols.append('month')
                    if df.month.max() > 12:
                       df.month[df.month > 12] -= 12
                if self.args.use_week == 1:
                    cols.append('week')
                    df['week'] = [d.week for d in df.time] # pd.to_datetime()
                if self.args.use_4hour == 1:
                    df['hour'] = df.hour // 4
            return df[cols + ['value']]
            # return df.groupby(cols)['value'].mean().reset_index()

        self.init = read_csv(self.prefix+"_init.csv").values.astype(np.float64)
        self.pool = read_csv(self.prefix+"_pool.csv").values.astype(np.float64)
        self.valid = read_csv(self.prefix+"_valid.csv").values.astype(np.float64)
        self.test = read_csv(self.prefixtest + "_test.csv").values.astype(np.float64)
        self.testidx = None

        def get_uniq(d):
            return np.unique(d, axis=0) if len(d) else d
        self.loc_init = get_uniq(self.init[:, :2])
        self.loc_pool = get_uniq(self.pool[:, :2])

        # Merge init to pool
        self.pool = np.concatenate((self.init, self.pool))
        self.loc_pool = np.concatenate((self.loc_init, self.loc_pool))

        self.GP, dists = self._make2Dgraph(self.loc_pool)
        if self.args.baseline is not None:
            if 1: # Lerner
                llmin0, llmin1, llmax0, llmax1 = self.loc_pool[:, 0].min(), self.loc_pool[:, 1].min(), self.loc_pool[:, 0].max(), self.loc_pool[:, 1].max()
                Fmin0 = self.loc_pool[:, 0] > llmin0
                Fmin1 = self.loc_pool[:, 1] > llmin1
                Fmax0 = self.loc_pool[:, 0] < llmax0
                Fmax1 = self.loc_pool[:, 1] < llmax1
                F = Fmin0.astype(int) + Fmin1.astype(int) + Fmax0.astype(int) + Fmax1.astype(int)
                self.lerner = torch.tensor(F.reshape(1, -1))
            if 1: # Coverage
                self.allpairdist = torch.tensor(dists)

        self.adjGP = nx.adjacency_matrix(self.GP)
        self.normadjGP = preprocess_adj(self.adjGP)
        self._toTensor()

        def make_map(loc, data):
            map, mask = [], []
            for i in range(len(loc)):
                F = np.all(data[:, :2] == loc[i], axis=1)
                if self.args.predictforecast:
                    T = data[:, 2]
                    F2 = T <= (T.max()-self.args.predictforecast)
                    F = np.logical_and(F,F2)
                mask.append(F)
                map.append(F.nonzero()[0])
            if not self.args.predictforecast:
                assert np.sum([len(map[i]) for i in range(len(map))]) == data.shape[0]
            return map, mask

        self.GPmap = make_map(self.loc_pool, self.pool)
        self.init = torch.from_numpy(self.init).to(self.device)
        self.pool = torch.from_numpy(self.pool).to(self.device)
        self.valid = torch.from_numpy(self.valid).to(self.device)
        self.test = torch.from_numpy(self.test).to(self.device)

        if self.args.predictforecast:
            def get_flag(data):
                return data[:, 2] > (data[:, 2].max() - self.args.predictforecast)
            self.pool_flag = get_flag(self.pool)
            self.valid_flag = get_flag(self.valid)
            self.test_flag = get_flag(self.test)

        self.X = self.pool[:,:-1]
        self.X = self.X.expand([self.args.batchsize] + list(self.X.size()))
        self.Y = self.pool[:,-1]
        # self.Y = self.Y.expand([self.args.batchsize] + list(self.Y.size()))
        self.normadj = self.YV = self.Y

    def _registerStat(self):
        L = OrderedDict()
        L["name"] = self.name
        if self.undirected is None:
            L["nnode"] = len(self.pool)
            L["nnodeGP"] = len(self.loc_pool)
            L["nedge"] = 0
            L["nedgeGP"] = 0
        else:
            L["nnode"] = self.G.number_of_nodes()
            L["nnodeGP"] = self.GP.number_of_nodes()
            L["nedge"] = self.G.number_of_edges()
            L["nedgeGP"] = self.GP.number_of_edges()
        L["nfeat"] = self.X.shape[1]
        L["nclass"] = self.Y.max() + 1
        L["sparse"] = self.sparse
        L.update(self.bestconfig)
        self.stat = L

    def printStat(self):
        logdicts(self.stat,tablename="dataset stat")

    def _getNormDeg(self):
        if self.sparse:
            self.deg = torch.sparse.sum(self.adj, dim=1).to_dense()
        else:
            self.deg = torch.sum(self.adj, dim=1)
        normdeg = self.deg / self.deg.max()
        self.deg = normdeg.expand([self.args.batchsize]+list(normdeg.size()))

    def makeValTestMask(self, args, fix_test=False):
        print('Splitting Validation/Test Data')
        assert args.resplit < 0
        with open(self.prefix + '.t.txt', 'r') as f:
            L = f.readlines()
        valid_,testid_ = [],[]
        if len(L) == 1:
            L = [[int(e) for e in x.strip().split()] for x in L]
            L = L[0]
            valid_ = [i for i in range(L[0],L[1])]
            testid_.append([i for i in range(L[2],L[3])])
            testid_.append([i for i in range(L[4],L[5])])
        elif len(L) == 2:
            L = [[int(e) for e in x.strip().split()] for x in L]
            testid_ = [np.sort(L[0][1:]).tolist()]
            valid_ = np.sort(L[1][1:]).tolist()
        else:
            L = [[int(e) for e in x.strip().split()] for x in L[:2]]
            for idx in range(0,len(L[0]),2):
                valid_ += [i for i in range(L[0][idx],L[0][idx+1])]
            for idx in range(0, len(L[1]), 2):
                testid_.append([i for i in range(L[1][idx], L[1][idx + 1])])
        print('Vald/Test Len:', len(valid_), len(testid_))

        self.valtestrng = [(0,1), (1,len(testid_)+1)]
        self.valtestid, self.valtestlabel, self.valtestpmval, self.valtestlabelcpu = [], [], [], []
        for testid in [valid_] + testid_:
            self.valtestid.append(torch.tensor(testid).to(self.device))
            self.valtestlabel.append(self.Y[testid].to(self.device))
            self.valtestlabelcpu.append(self.Y[testid].cpu())
            if self.args.lossfn == 2:
                self.valtestpmval.append(self.YV[testid].to(self.device))
        if self.GP:
            self.skip_nodes(args)

    def skip_nodes(self,args):
        print('Removing test+valids', sum([len(v) for v in self.GPmap]))
        if not self.GPmap2:
            from copy import deepcopy
            self.GPmap2 = deepcopy(self.GPmap)
            nodes = torch.cat((self.testid[0], self.valid[0]), axis=0)
            skip = []
            for n, i in zip(nodes, torch.tensor(self.GPidx)[nodes]):
                self.GPmap2[i].remove(n)
                if not len(self.GPmap2[i]):
                    skip.append(i)
            self.GPskip = skip
            # Done with conversion, free mem
            del self.GPidx

        self.maskGP = torch.zeros((self.args.batchsize, self.stat['nnodeGP'])).to(torch.float).to(self.device)
        for x in range(self.args.batchsize):
            self.maskGP[x, self.GPskip] = 1
        del self.GPskip
        print('Removed  test+valids', sum([len(v) for v in self.GPmap2]), 'Masking', int(sum(self.maskGP[0]).item()),'out of',self.maskGP.shape[1])

    def process(self,args):
        if self.undirected is None:
            return None

        if int(self.bestconfig['feature_normalize']):
            self.X = column_normalize(preprocess_features(self.X)) # take some time

        self.normadj = preprocess_adj(self.adj)
        if self.GP:
            self.normadjGP = preprocess_adj(self.adjGP)

        if not self.sparse:
            self.adj = self.adj.todense()
            self.normadj = self.normadj.todense()
        self._toTensor()

        self.makeValTestMask(args)

        # Free unused vars
        del self.adj
