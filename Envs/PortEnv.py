
import pandas as pd
import numpy as np
from gym.utils import seeding
from gym import spaces

class PortEnv:
    def __init__(self, data, tickers,init_invest, period):
        self.data = data
        self.tickers = tickers
        if len(tickers) != self.data.shape[1]:
            raise ValueError("tickers...")
        self.period = period
        self.init_invest = init_invest
        
        self.cur_step = None
        self.n_step = self.data.shape[0]//self.period
        self.weights = None
        self.cash = None
        self.stocks_onwed = None

        self.action_space = spaces.Box(0.0,1.0,shape=(self.data.shape[1],))
        self.observation_space = spaces.Dict({
            "history": spaces.Box(
                -10, 
                1,
                (self.data.shape[0], self.data.shape[1], 3)#Tror jag
            ),
            "weights":self.action_space
        })
        print(self.observation_space)

        self._seed()
        self.reset()


    def _initial_weights(self):
        weights = {}
        for ticker in self.tickers:
            weights[ticker] = 1/self.data.shape[1]
        return weights
    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.cur_step = 0
        self.weights = self._initial_weights()
        self.cash = self.init_invest
        self.stocks_onwed = {s: 0 for s in self.tickers}
        return self._get_obs()
    def _get_obs(self):
        """
        Ska returna de gamla weightsen och log returns
        """
        d = self.data[:self.period*(self.cur_step+1), :]
        r = np.diff(np.log(d))
        return r, self.weights
    def step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        self._trade(action)
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step -1
        info = {"cur_val":cur_val}
        return self._get_obs(), reward, done, info
    def _get_val(self):
        s = np.sum(list(self.stocks_onwed.values())*self.data[self.cur_step,:])+self.cash
        return s
    def _buy_until_target(self, ticker, i, action):
        cur_stock_price = self.data[self.cur_step, i]
        cur_weight = self.cash*self.weights[ticker]
        target_weight = self.cash*action
        
        while cur_weight < target_weight:
            cur_weight += cur_stock_price
            self.stocks_onwed[ticker]+=1
    def _sell_until_target(self, ticker, i, action):
        cur_stock_price = self.data[self.cur_step, i]
        cur_weight = self.cash*self.weights[ticker]
        target_weight = self.cash*action
        while cur_weight > target_weight:
            cur_weight -= cur_stock_price
            self.stocks_onwed[ticker]-=1

    def _trade(self, action):
        """
        action - nya vektorn med weights
        logik:
            1. Ta skillnad mellan ny och gammal vikt
            2. Köp/sälj så att den ligger på target weight
        """
        i = 0
        for ticker, weight in self.weights.items():
            diff = weight - action[i]
            
            if diff > 0:
                self._buy_until_target(ticker,i, action[i])
            elif diff < 0:
                self._sell_until_target(ticker, i, action[i])
            self.weights[ticker] = action[i]
            i+=1






    
