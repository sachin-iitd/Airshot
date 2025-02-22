import numpy as np
from src.RLselect.common import *


class RewardShaper(object):
    def __init__(self,args):
        self.metric = args.metric
        self.shaping = args.shaping
        self.entcoef = args.entcoef
        self.gamma = 0.5
        self.rate = 0.05
        self.alpha = args.frweight # to make tune the ratio of finalreward and midreward
        self.historymean,self.histrvar = None,None
        pass

    def reshape(self,rewards_all,finalrewards_all,lossreward,logprobs):
        rewards_sub, finalrewards = self._roughProcess(rewards_all,finalrewards_all,lossreward)
        self.componentRatio(rewards_sub,finalrewards,logprobs)
        rewards = np.zeros_like(rewards_sub)
        if "0" in self.shaping:
            rewards += rewards_sub
        if "1" in self.shaping:
            for i in range(rewards_sub.shape[0]-1,0,-1):
                rewards_sub[i-1] += self.gamma*rewards_sub[i]
            rewards += rewards_sub
        if "2" in self.shaping: #2
            rewards += finalrewards*self.alpha

        if "3" in self.shaping:
            if self.historymean is None:
                self.historymean = rewards.mean(1,keepdims=True)
            else:
                self.historymean = self.historymean * (1 - self.rate) + self.rate * rewards.mean(1,keepdims=True)
            rewards = rewards - self.historymean

        if "4" in self.shaping:
            if self.histrvar is None:
                self.histrvar = (rewards**2).mean(1,keepdims=True)
            else:
                self.histrvar = self.histrvar * (1 - self.rate) + self.rate * (rewards**2).mean(1,keepdims=True)
            if np.sum(rewards):
                rewards = rewards/(np.power(self.histrvar,0.5))
        return rewards

    def _roughProcess(self,rewards_all,finalrewards_all,lossreward):
        mic,mac,rmse = [np.array(x) for x in list(zip(*rewards_all))]
        finalmic,finalmac, finalrmse = [np.array(x) for x in finalrewards_all]
        lossreward = -np.array(lossreward)
        if self.metric == "microf1":
            rewards = mic
            finalrewards = finalmic
        elif self.metric == "macrof1":
            rewards = mac
            finalrewards = finalmac
        elif self.metric == "mix":
            rewards = (mic+mac)/2
            finalrewards = (finalmic+finalmac)/2
        elif self.metric == "mse":
            rewards = lossreward[:-1,:]
            finalrewards = lossreward[-1,:]
        elif self.metric == "rmse":
            rewards = -rmse
            finalrewards = -finalrmse
        else:
            raise NotImplementedError("metric <{}> is not implemented".format(self.metric))
        rewards_sub = rewards[1:,:]-rewards[:-1,:]
        return rewards_sub,finalrewards

    def componentRatio(self,rewards_sub,finalrewards_all,logprobs):
        r_mean = np.mean(np.abs(rewards_sub))
        if r_mean < 1e-10:
            return
        f_mean = np.mean(finalrewards_all)
        lp_mean = np.mean(np.abs(logprobs))
        f_ratio = f_mean/r_mean*self.alpha
        lp_ratio = lp_mean/r_mean*self.entcoef
        getLogger().debug("rmean {:.4f},fratio {:.2f}x{:.4f}={:.3f}, "
                    "lpratio {:.1f}x{:.5f}={:.3f}".format(r_mean,
                                                           f_mean/r_mean,self.alpha,f_ratio,
                                                           lp_mean/r_mean,self.entcoef,lp_ratio))