# https://github.com/sachin-iitd/AirShot

import numpy as np
import time
import random
import os

import torch
import argparse
from torch.distributions import Categorical
import torch.nn.functional as F
from src.RLselect.dataloader import GraphLoader
from src.RLselect.player import Player
from src.RLselect.env import Env
from src.RLselect.rewardshaper import RewardShaper
from src.RLselect.policynet import *
from src.RLselect.common import *
from src.RLselect.utils import *
from src.baselines.mutualinfo import mi_placement
from src.baselines.random import random_placement
from src.baselines.pagerank import pagerank_placement
from src.baselines.bwcentrality import bwcentrality_placement
from src.baselines.degree import degree_placement
from src.baselines.maxerr import maxerr_placement
from src.baselines.coverage import coverage_placement
from src.baselines.lerner import lerner_placement

switcher = {'gcn':PolicyNet,
            'RLselect':PolicyNetSageMax,
            }
dataset = None
baselines = [None,'random','bwcentrality','degree','pagerank','maxerr','coverage','lerner','mi']
dumpcsvtest = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default=None)
    parser.add_argument("--seasons",type=str,default=['ma,so,dj','{}'][1])
    parser.add_argument("-cmntest", action="store_true", default=True)
    parser.add_argument("-use_month", action="store_true", default=False)
    parser.add_argument("-use_week", action="store_true", default=False)
    parser.add_argument("-use_day", action="store_true", default=False)
    parser.add_argument("-use_hour", action="store_true", default=False)
    parser.add_argument("-use_4hour", action="store_true", default=False)
    parser.add_argument("--nhid",type=str,default='64')
    parser.add_argument("--pnhid",type=str,default='8+8')
    parser.add_argument("--dropout",type=float,default=0.2)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--lr",type=float,default=3e-2)
    parser.add_argument("--rllr",type=float,default=1e-2)
    parser.add_argument("--entcoef",type=float,default=0)
    parser.add_argument("--frweight",type=float,default=1e-3)
    parser.add_argument("--batchsize",type=int,default=2)

    parser.add_argument("--sensors",type=int,default=0,help="num static sensors in region")
    parser.add_argument("--budget",type=int,default=10,help="budget per class")
    parser.add_argument("--metric",type=str,default="rmse")
    parser.add_argument("--outfile",type=str,default=None)

    parser.add_argument("--remain_epoch",type=int,default=1,help="continues training $remain_epoch"
                                                                  " epochs after all the selection")
    parser.add_argument("--shaping",type=str,default="234",help="reward shaping method, 0 for no shaping;"   # 234
                                                              "1 for add future reward,i.e. R= r+R*gamma;"
                                                              "2 for use finalreward;"
                                                              "3 for subtract baseline(value of curent state)"
                                                              "1234 means all the method is used,")
    parser.add_argument("--logfreq",type=int,default=1)
    parser.add_argument('--accrange', type=int, default=100)
    parser.add_argument("--maxepisode",type=int,default=0)
    parser.add_argument("--save",type=int,default=0)
    parser.add_argument("--savename",type=str,default="")
    parser.add_argument("--policynet",type=str,default='RLselect')
    parser.add_argument("--predictnet",type=str,default=['gcn','gcnn','rf'][-1])
    parser.add_argument("--predictforecast",type=int,default=0)

    parser.add_argument("--baseline",type=str,default=baselines[0])
    parser.add_argument("-seqact", action="store_true", default=False)
    parser.add_argument("-nosample", action="store_true", default=False)
    parser.add_argument("--use_entropy",type=int,default=0)
    parser.add_argument("--use_degree",type=int,default=0)
    parser.add_argument("--use_local_diversity",type=int,default=1)
    parser.add_argument("--use_select",type=int,default=1)
    parser.add_argument("--use_pagerank",type=int,default=1)
    parser.add_argument("--use_rmse",type=int,default=1)
    parser.add_argument("--weight",type=int,default=3)
    parser.add_argument("-seppolicygraph", action="store_true", default=True)

    parser.add_argument('--pg', type=str, default='reinforce')
    parser.add_argument('--ppo_epoch', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--schedule', type=int, default=0)

    parser.add_argument("-dumpcsv", action="store_true", default=True)
    parser.add_argument("-dumppredict", action="store_true", default=False)
    parser.add_argument("-dumpnodes", action="store_true", default=True)
    parser.add_argument("-dumpmetrics", action="store_true", default=True)
    parser.add_argument("--dsfolder",type=str,default=None)
    parser.add_argument('--sageGCN', type=int, default=0)
    parser.add_argument('--lossfn', type=int, default=2)    # [nll_softmax, mse_sig, mse_softmax]
    parser.add_argument("--classify", type=int, default=0)
    parser.add_argument('--resplit', type=int, default=-1)  # -1 means take pre-split distribution
    parser.add_argument('--onehot', type=int, default=1)

    parser.add_argument("--suffix",type=str,default="")
    parser.add_argument("--model_dir",type=str,default=None)
    parser.add_argument("--test",type=str,default=None)

    parser.add_argument("-debug", action="store_true", default=False)
    parser.add_argument('--testgraphprefix', type=str, default=None)
    parser.add_argument('--testgraphstart', type=int, default=0)

    parser.add_argument('--undirected', type=str, default=None)

    args = parser.parse_args()

    if args.debug:
        # args.use_month = True
        # args.use_week = True
        # args.use_4hour = True
        # args.use_day = True
        args.use_hour = True
        args.dataset = 'kk1'
        if args.baseline is None:
            args.baseline = baselines[0]  # SELECT Baseline
        if len(args.suffix) == 0:
            args.suffix = '1'
        if args.suffix[0] != '_':
            args.suffix = '_' + args.suffix
        args.budget = 5
        if args.baseline:
            args.outfile = '1'
        args.dumpcsv = '1'

    if args.dataset is None:
        print('--dataset is must !!!')
        exit(1)

    args.nhid = [int(n) for n in args.nhid.split('+')] if len(args.nhid) else []
    args.pnhid = [int(n) for n in args.pnhid.split('+')]

    global dataset
    dataset = args.dataset
    args.seasons = args.seasons.split(',')
    args.dataset = args.dataset.format(args.seasons[0] if len(args.seasons)==1 else '[]')

    if args.test:
        args.model_dir = '/'.join(args.test.split('/')[:-2])

    if args.predictnet and args.predictnet.lower().startswith('sage'):
        args.sageGCN = 1

    if args.maxepisode <= 0:
        if args.baseline == 'random':
            args.accrange = args.maxepisode = 3
        elif args.test or args.baseline:
            args.accrange = args.maxepisode = 1
        else:
            args.maxepisode = 1000

    if args.outfile == '1':
        if args.test:
            args.outfile = 'out/test_{}.{}.txt'.format(args.dataset,args.test.split('/')[-1])
        elif args.baseline:
            args.outfile = 'out/base_{}.txt'.format(args.dataset)
    if args.dumpcsv == '1':
        global dumpcsvtest
        file = '{}_{}{}'.format(args.baseline if args.baseline else args.policynet, args.dataset, args.suffix)
        args.dumpcsv = 'out/' + file
        dumpcsvtest = 'out/{}/'.format(args.dataset)
        if args.baseline is None and not args.seqact and not os.path.exists(dumpcsvtest):
            os.makedirs(dumpcsvtest)
        dumpcsvtest = dumpcsvtest + file

    return args

def improve_suffix(args):
    suffix = '_'
    if '_B' not in args.suffix:
        suffix += 'B{}_'.format(args.batchsize)
    if '_N' not in args.suffix:
        suffix += 'N{}_'.format(args.budget)
    if args.sageGCN and '_G' not in args.suffix:
        suffix += 'G{}_'.format(args.sageGCN)
    if args.baseline:
        suffix += 'P{}_'.format(args.baseline)
    if '_P' not in args.suffix and args.baseline is None:
        suffix += 'P{}_'.format(args.policynet)
    if '_R' not in args.suffix and args.predictnet:
        suffix += 'R{}{}_'.format(args.predictnet,len(args.nhid))
    if '_S' not in args.suffix:
        state = ''
        for i,s in enumerate([args.use_degree,args.use_pagerank,args.use_entropy,args.use_rmse,args.use_local_diversity,args.use_select]):
            state += 'DPERNI'[i] if s else ''
        suffix += 'S{}_'.format(state)
    if '_V' not in args.suffix:
        suffix += 'V{}_'.format(args.remain_epoch)
    return suffix

class SingleTrain(object):

    def __init__(self, args):

        if args.gpu >= 0 and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            self.device = torch.device("cuda:"+str(args.gpu))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
            self.device = "cpu"

        if args.model_dir is None:
            our_suffix = improve_suffix(args)
            randid = random.randint(1, 99)
            time_suffix = time.strftime('_T_%m%d_%H%M%S', time.localtime(time.time()))+'_{:02d}'.format(randid)
            basedir = 'models/'
            args.model_dir = basedir + args.dataset + our_suffix + args.suffix + time_suffix
            if not os.path.exists(basedir):
                os.mkdir(basedir)
            os.mkdir(args.model_dir)
            os.mkdir(args.model_dir+'/models')

        self.startepisode = 1
        self.args = args

        self.graphs, self.players, self.rshapers = [], [], []
        # Set dataset
        for idx,ds in enumerate(args.seasons):
            g = GraphLoader(dataset, idx=idx, undirected=args.undirected, sparse=True,args=args, folder=args.dsfolder, device=self.device)
            logargs(args, tablename="config")
            g.process(args)
            self.graphs.append(g)
            p = Player(g, args, device=self.device).to(self.device)
            self.players.append(p)
            self.rshapers.append(RewardShaper(args))

        self.env = Env(self.players,args,self.device)
        self.policy = switcher[args.policynet](self.args,self.env.statedim).type(torch.DoubleTensor).to(self.device)
        print('PolicyNet: {} {} {}'.format(self.env.statedim, args.baseline if args.baseline else args.policynet, '' if args.baseline else self.policy.__class__.__name__))
        if args.test:
            from src.RLselect.query import ProbQuery
            self.policy.load_state_dict(torch.load("{}.pkl".format(args.test)))
            self.query = ProbQuery("hard")

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.args.rllr)
        if self.args.schedule:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [1000, 3000], gamma=0.1, last_epoch=-1)
        if args.outfile or 1:
            self.accmetertest3std = AverageMeter("aveaccmetertest3std",ave_step=args.accrange)
            self.accmetertestrmse = AverageMeterSimple()
            self.accmeterteststd = AverageMeterSimple()
        L = len(args.seasons)+1
        self.bestrmsevald, self.bestrmsetest, self.beststdtest, self.bestepisode = [bigint] * L, [bigint] * L, [bigint] * L, [-1] * L
        self.action_index = torch.zeros([self.args.batchsize, self.args.budget]).type(torch.LongTensor).to(self.device)

        test_extn,mode = ('.test','a') if args.test else ('','w')
        if args.dumpnodes:
            self.dumpnodes = open(args.model_dir + '/nodes.txt' + test_extn, mode)
        if args.dumpmetrics:
            self.dumpmetrics = open(args.model_dir + '/metrics.txt' + test_extn, mode, buffering=1)
            import sys
            print(*sys.argv[1:], file=self.dumpmetrics)
            print(args, file=self.dumpmetrics)
        if args.dumppredict:
            self.dumppredict = open(args.model_dir + '/classify.txt' + test_extn, mode)

    def jointtrain(self):

        #for episode in range(1,maxepisode+1):
        episode = self.startepisode
        for _ in range(self.startepisode, self.args.maxepisode+1):
            self.all_actions = []
            self.met_final, self.met_test = [], []
            for playerid in range(len(self.players)):
                self.playOneEpisode(episode, playerid)
                if self.bestepisode[playerid]==episode: # or (not (episode % 100)):
                    pathname = "/models/{}{}{}_{}".format(self.args.policynet,self.args.savename,playerid,episode)
                    torch.save(self.policy.state_dict(),self.args.model_dir+pathname+'.pkl')
            episode += 1

        # Exiting, mark best model
        if not args.test:
            for playerid in range(len(self.players)):
                open('{}/{}.best{}'.format(args.model_dir,self.bestepisode[playerid],playerid),'w')
        logargs(self.args, tablename="config")

    def playOneEpisode(self, episode, playerid=0):

        self.playerid = playerid
        self.env.reset(playerid,resplit=True if args.resplit>0 else False)
        rewards, rewardpool, logp_actions, p_actions, rewardtest, lossreward = [], [], [], [], [], []
        self.states, self.actions, self.pools = [], [], []
        self.entropy_reg = []
        self.dbg_ctr = 0

        r, r2, l, rp = self.env.step(playerid=playerid)
        rewards.append(r)  # From validation data
        rewardtest.append(r2)  # From test data
        rewardpool.append(rp)  # From pool-train data
        lossreward.append(l)  # From training loss

        testoutfile = None
        if args.dumpcsv:
            if args.baseline is None and not args.seqact:
                testoutfile = open(dumpcsvtest + '.' + str(episode) + '.test.csv', 'w')
                print('episode', 'epoch', 'batch', 'sensor', 'lat', 'long', 'pid',
                      *[i for i in range(len(self.players[playerid].pred_vals[0]))], sep=',', file=testoutfile)
            elif episode == 1 and playerid == 0:
                testoutfile = open(args.dumpcsv + '.test.csv', 'w')
                print('episode', 'epoch', 'batch', 'sensor', 'lat', 'long', 'pid',
                      *[i for i in range(len(self.players[playerid].pred_vals[0]))], sep=',', file=testoutfile)
            else:
                testoutfile = open(args.dumpcsv + '.test.csv', 'a')
            # At two places
            for b in range(self.args.batchsize):
                print(episode, 0, b, -1, -1, -1, playerid, *self.players[playerid].pred_vals[b].round(decimals=2).tolist(), sep=',', file=testoutfile)

        mi = None
        if args.baseline == 'mi':
            mi = mi_placement(self.players[self.playerid].G, self.args.budget)

        for epoch in range(self.args.budget):
            state = self.env.getState(playerid)
            pool = self.env.players[playerid].getPool(reduce=False)
            if self.args.pg == 'ppo':
                self.states.append(state)
                self.pools.append(pool)

            logits = self.policy(state,self.graphs[playerid].normadjGP)
            action,logp_action, p_action = self.selectActions(logits,pool)

            pool = pool[1].reshape(self.args.batchsize, -1)
            if args.seqact:
                action = pool[[x for x in range(self.args.batchsize)], (episode-1,)]
            elif args.test:
                action = self.query(logits, pool)
            elif args.baseline is None: # RLselect
                action = action.detach()
            elif args.baseline == 'random':
                action = random_placement(self.args.batchsize,pool)
            elif args.baseline == 'pagerank':
                action = pagerank_placement(self.args.batchsize,pool,state[:,:,-1])
            elif args.baseline == 'bwcentrality':
                action = bwcentrality_placement(self.args.batchsize,pool,self.env.players[playerid].G.bwcentrality)
            elif args.baseline == 'degree':
                action = degree_placement(self.args.batchsize,pool,self.env.players[playerid].G.deg[0])
            elif args.baseline == 'maxerr':
                action = maxerr_placement(self.args.batchsize,pool,state[:, :, 0])
            elif args.baseline == 'coverage':
                action = coverage_placement(self.args.batchsize,pool,self.env.players[playerid].G.allpairdist)
            elif args.baseline == 'lerner':
                action = lerner_placement(self.args.batchsize,pool,self.env.players[playerid].G.lerner[0])
            elif args.baseline == 'mi':
                for i in range(self.args.batchsize):
                    action[i] = mi[epoch]
            elif args.baseline:
                    raise 'Unknown baseline: ' + args.baseline

            for i in range(self.args.batchsize):
                self.action_index[i, epoch] = action[i]
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            r,r2,l,rp = self.env.step(action, playerid)
            rewards.append(r)  # From validation data
            rewardtest.append(r2)  # From test data
            rewardpool.append(rp)  # From pool-train data
            lossreward.append(l)  # From training loss

            self.entropy_reg.append(-(self.valid_probs * torch.log(1e-6 + self.valid_probs)).sum(dim=1) / np.log(self.valid_probs.size(1)))

            if args.dumpcsv:
                # At two places
                # with open(args.dumpcsv + '.test.csv', 'a') as f:
                for b in range(self.args.batchsize):
                    print(episode, epoch+1, b, action[b].item(), *self.graphs[self.playerid].loc_pool[action[b]], playerid, *self.players[playerid].pred_vals[b].round(decimals=2).tolist(), sep=',', file=testoutfile)

        losslast, lossavg = self.env.players[playerid].trainRemain()
        self.entropy_reg = torch.stack(self.entropy_reg).to(self.device)
        finalrewards = self.env.players[playerid].validation(rerun=True)
        testrewards = self.env.players[playerid].validation(rerun=False,test=True)
        # rewards.append(finalrewards)
        # rewardtest.append(testrewards)
        # rewardpool.append(self.env.players[playerid].validation(test=-1))
        lossreward.append(losslast)
        self.all_actions.append(self.action_index.clone())

        micfinal = mean_std(finalrewards[0])[0]*100
        mictest = mean_std(testrewards[0])[0]*100
        macfinal = mean_std(finalrewards[1])[0]*100
        mactest = mean_std(testrewards[1])[0]*100

        def proc_metrics(finalrewards=None, testrewards=None, playerid=-1):
            if len(self.players) > 1:
                if playerid < 0:
                    # Merge the metrics
                    testrewards = np.array(self.met_test).mean(axis=0)
                    finalrewards = np.array(self.met_final).mean(axis=0)
                else:
                    self.met_final.append(finalrewards)
                    self.met_test.append(testrewards)

            rmsestdvald = mean_std(finalrewards[2])
            rmsevald, stdvald = rmsestdvald[0], rmsestdvald[1]
            rmsestdtest = []
            for i in range(0,len(testrewards[2]),args.batchsize):
                rmsestdtest.append(mean_std(testrewards[2][i:i+args.batchsize]))
            if 1:
                batchrmse, rmsetest, stdtest = '', 0.0, 0.0
                for rmsestd in rmsestdtest:
                    batchrmse += '{:.1f} {:.1f} '.format(rmsestd[0], rmsestd[1])
                    rmsetest += rmsestd[0]
                    stdtest += rmsestd[1]
                rmsetest, stdtest = rmsetest / len(rmsestdtest), stdtest / len(rmsestdtest)
            avgrmse = '{:.1f} {:.1f} {:.1f} {:.1f}'.format(rmsevald, stdvald, rmsetest, stdtest)
            if rmsevald < self.bestrmsevald[playerid]:
                self.bestepisode[playerid] = episode
                self.bestrmsevald[playerid] = rmsevald
                self.bestrmsetest[playerid] = rmsetest
                self.beststdtest[playerid] = stdtest
            return batchrmse, avgrmse, rmsevald, rmsetest, stdtest, rmsestdtest

        batchrmse, avgrmse, rmsevald, rmsetest, stdtest, rmsestdtest = proc_metrics(finalrewards, testrewards, playerid)

        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        shapedrewards = self.rshapers[playerid].reshape(rewards,finalrewards,lossreward,logp_actions.detach().cpu().numpy())

        loss = 0
        if episode > 10:
            loss = self.finishEpisode(shapedrewards, logp_actions, p_actions, self.action_index, logits, playerid)

        if args.dumppredict:
            a = np.array(list(rewardtest)).T
            print(episode, *np.round(a[0][0],2), file=self.dumppredict)
            print(epoch, *np.round(a[0][1],2), file=self.dumppredict)

        if episode % self.args.logfreq == 0:
            print("Ep {}-{}: ClsLoss: {:.2f} | Acc: Valid {:.1f}, Test {:.1f}/{:.1f}/{:.1f} | Best {} {:.1f} {:.1f}({:.1f})"
                  .format(episode, playerid, sum(losslast), rmsevald, mictest, mactest, rmsetest,
                          self.bestepisode[playerid], self.bestrmsevald[playerid], self.bestrmsetest[playerid], self.beststdtest[playerid])
                    )
            if not (episode % 40) and playerid==0 and self.device != 'cpu':
                print(args.model_dir.split('/')[-1], 'Gpu:', args.gpu)

        if self.args.dumpnodes:
            print(episode, *(self.action_index.shape), *(self.action_index.cpu().numpy().reshape(-1)), file=self.dumpnodes)
        if self.args.dumpmetrics:
            print(episode, '{:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(sum(losslast), lossavg, loss, micfinal, macfinal, mictest, mactest),
                  batchrmse, avgrmse, file=self.dumpmetrics)
        if args.outfile or 1:
            self.accmetertest3std.update(stdtest)
            for rmsestd in rmsestdtest:
                self.accmetertestrmse.update(rmsestd[0])
                self.accmeterteststd.update(rmsestd[1])
            if episode >= self.args.maxepisode:
                if args.test or args.baseline:
                    fmt = '{} {} {} {:.1f} ({:.1f}) {:.1f} ({:.1f})'.format(args.dataset, args.baseline, self.args.budget, -1,self.accmetertest3std(),self.accmetertestrmse(),self.accmeterteststd())
                    print('Avg:', fmt)
                    if args.outfile:
                        print(fmt, file=open(args.outfile,'a'))

        if args.dumpcsv:
            def dumpcsv(playerid=-1):
                _rewardpool, _rewards, _rewardtest = rewardpool, rewards, rewardtest
                if playerid == 0:
                    self.met_dict = {'pool':[], 'valid':[], 'test':[]}
                elif playerid<0:
                    _rewardpool = np.array(self.met_dict['pool']).mean(axis=0)
                    _rewards = np.array(self.met_dict['valid']).mean(axis=0)
                    _rewardtest = np.array(self.met_dict['test']).mean(axis=0)
                with open(args.dumpcsv+'.csv', 'aw'[episode==1 and playerid==0]) as f:
                    if playerid >= 0:
                        f2 = open(args.dumpcsv+'.nodes.csv', 'aw'[episode==1 and playerid==0])
                    for typ,rew in [('pool',_rewardpool),('valid',_rewards),('test',_rewardtest)]:
                        if playerid < 0:
                            a = rew
                        else:
                            a = np.array(list(rew)).T
                            self.met_dict[typ].append(a)
                        for i in range(args.batchsize):
                            print(episode, playerid, self.bestepisode[playerid], epoch+1, typ, i, *a[i][-1], sep=',', file=f)
                            if typ == 'pool' and playerid>=0:
                                loc = self.graphs[0].loc_pool[self.action_index.cpu()]
                                print(episode, playerid, self.bestepisode[playerid], epoch+1, typ, i, *loc[i].reshape(-1), sep=',', file=f2)
                        if typ == 'test':
                            for i in range(args.batchsize, len(a)):
                                print(episode, playerid, self.bestepisode[playerid], epoch + 1, typ+str(i//args.batchsize), i%args.batchsize, *a[i][-1], sep=',', file=f)

            dumpcsv(playerid)
            # Dump merged metrics
            if len(self.players) > 1 and playerid == len(self.players)-1:
                batchrmse, avgrmse, rmsevald, rmsetest, stdtest, rmsestdtest = proc_metrics()
                dumpcsv()
                print("Ep {}-{}: ClsLoss: {:.2f} | Acc: Valid {:.1f}, Test {:.1f}/{:.1f}/{:.1f} | Best {} {:.1f} {:.1f}({:.1f})"
                      .format(episode, 'M', sum(losslast), rmsevald, mictest, mactest, rmsetest,
                              self.bestepisode[playerid], self.bestrmsevald[playerid], self.bestrmsetest[playerid], self.beststdtest[playerid]))

        return micfinal,macfinal,-rmsevald

    def finishEpisode(self,rewards,logp_actions, p_actions, actions, logits, playerid=0):

        rewards = torch.from_numpy(rewards).type(torch.float32).to(self.device)

        if (self.args.pg == 'reinforce'):
            #losses =torch.sum(-logp_actions*rewards,dim=0)
            #loss = torch.mean(losses) - self.args.entcoef * self.entropy_reg.sum(dim=0).mean()
            losses = logp_actions * rewards
            if self.args.entcoef > 1e-10:
                losses += self.args.entcoef * self.entropy_reg
            loss = -torch.mean(torch.sum(losses, dim=0))
            self.opt.zero_grad()
            # for x in [loss, losses, self.entropy_reg]: #, sim2]:
            #     x.retain_grad()
            loss.backward()
            self.opt.step()
            if self.args.schedule:
                self.scheduler.step()
        elif (self.args.pg == 'ppo'):
            epsilon = 0.2
            p_old = p_actions.detach()
            r_sign = torch.sign(rewards).type(torch.float32)
            for i in range(self.args.ppo_epoch):
                if (i != 0):
                    p_actions = [self.trackActionProb(self.states[i], self.pools[i], self.actions[i]) for i in range(len(self.states))]
                    p_actions = torch.stack(p_actions)
                ratio = p_actions / p_old
                losses = torch.min(ratio * rewards, (1 + epsilon * r_sign) * rewards)
                loss = -torch.mean(losses)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return loss.item()


    def trackActionProb(self, state, pool, action):
        logits = self.policy(state, self.graphs[self.playerid].normadj)
        valid_logits = logits[pool].reshape(self.args.batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)
        prob = valid_probs[list(range(self.args.batchsize)), action]
        return prob


    def selectActions(self,logits,pool):
        valid_logits = logits[pool].reshape(self.args.batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        valid_logits2 = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits2, dim=1)
        #if torch.sum(torch.isnan(valid_probs)):
        #    print("valid_probs", valid_probs)
        #    raise 'InVaid probs'
        self.valid_probs = valid_probs
        pool = pool[1].reshape(self.args.batchsize,-1)
        assert pool.size()==valid_probs.size()

        m = Categorical(valid_probs)
        if args.nosample:
            action_inpool = torch.argmax(valid_probs,axis=1)
        else:
            action_inpool = m.sample()
        if self.args.pg == 'ppo':
            self.actions.append(action_inpool)
        logprob = m.log_prob(action_inpool)
        prob = valid_probs[list(range(self.args.batchsize)), action_inpool]
        action = pool[[x for x in range(self.args.batchsize)],action_inpool]

        return action, logprob, prob


if __name__ == "__main__":

    args = parse_args()
    singletrain = SingleTrain(args)
    singletrain.jointtrain()
