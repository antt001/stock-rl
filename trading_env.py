import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Action space: 0 (Sell), 1 (Hold), 2 (Buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, owned_shares, current_price, other_features...]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3 + len(df.columns),))
        
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.owned_shares = 0
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        obs = np.array([
            self.balance,
            self.owned_shares,
            self.df.iloc[self.current_step][4],
            *self.df.iloc[self.current_step]
        ])
        return obs
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step][4]
        
        if action == 0:  # Sell
            shares_to_sell = self.owned_shares
            self.balance += shares_to_sell * current_price * (1 - self.transaction_fee_percent)
            self.owned_shares = 0
        elif action == 2:  # Buy
            affordable_shares = self.balance // current_price
            self.owned_shares += affordable_shares
            self.balance -= affordable_shares * current_price * (1 + self.transaction_fee_percent)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        
        portfolio_value = self.balance + self.owned_shares * current_price
        reward = portfolio_value - self.initial_balance
        
        return obs, reward, done, {}

# Usage example:
# df = pd.read_csv('stock_data.csv')
# env = TradingEnv(df)
