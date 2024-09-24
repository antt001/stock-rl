import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# For environment
from gym import Env
from gym.spaces import Discrete, Box


INITIAL_BALANCE = 10000

class TradingEnv(Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.total_steps = len(df)
        self.current_step = 0

        # Initialize state variables
        self.balance = INITIAL_BALANCE  # Starting cash
        self.shares_held = 0
        self.net_worth = self.balance
        self.prev_net_worth = self.net_worth
        self.max_net_worth = self.balance
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = Discrete(3)
        num_indicators = 13  # Number of features in the observation
        sample_obs = self._next_observation()
        # Observation space: Adjust according to your state representation
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32)

    def calculate_transaction_cost(self, action):
        if action in [1, 2]:  # Buy or Sell
            shares_traded = self.shares_held
            current_price = self.df.loc[self.current_step, 'Close']
            trade_value = shares_traded * current_price
            commission_rate = 0.001  # 0.1%
            transaction_cost = commission_rate * trade_value
        else:
            transaction_cost = 0
        return transaction_cost

    def calculate_risk(self):
        # Calculate drawdown
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        return drawdown

    def _next_observation(self):
        # Get the data points for the next state
        frame = np.array([
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'High'],
            self.df.loc[self.current_step, 'Low'],
            self.df.loc[self.current_step, 'Close'],
            self.df.loc[self.current_step, 'Volume'],
            # Add technical indicators if available
            self.df.loc[self.current_step, 'MA_Short'],
            self.df.loc[self.current_step, 'MA_Medium'],
            self.df.loc[self.current_step, 'MA_Long'],
            self.df.loc[self.current_step, 'BB_Upper'],
            self.df.loc[self.current_step, 'BB_Lower'],
        ])
        # Append additional state variables
        obs = np.append(
            frame, 
            [self.balance, self.shares_held, self.net_worth]
        )
        return obs.astype(np.float32)

    def step(self, action):
        # Save previous net worth
        self.prev_net_worth = self.net_worth

        # Execute one time step within the environment
        self.current_step += 1
        # done = self.current_step >= self.total_steps - 1

        # Calculate the reward
        reward = 0
        current_price = self.df.loc[self.current_step, 'Close']

        # Update net worth before action
        self.net_worth = self.balance + self.shares_held * current_price

        # Calculate stop-loss price
        if self.shares_held > 0:
            position_value = self.shares_held * current_price
            max_loss = self.net_worth * 0.05  # 5% of net worth
            stop_loss_price = self.entry_price - (max_loss / self.shares_held)
        else:
            stop_loss_price = None
        
        # Check if stop-loss is hit
        if stop_loss_price and current_price <= stop_loss_price:
            # Sell all shares held
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.entry_price = 0
            # print(f"Stop-loss triggered at price {current_price:.2f}")

    
        if action == 1:  # Buy
            # Determine position size based on risk management
            max_loss = self.net_worth * 0.05  # 5% of net worth
            stop_loss_distance = current_price * 0.05  # 5% below current price
            position_size = max_loss / stop_loss_distance

            # Ensure we don't buy more than we can afford
            shares_to_buy = min(
                position_size, self.balance // current_price
            )
            self.balance -= shares_to_buy * current_price
            self.shares_held += shares_to_buy
            self.entry_price = current_price  # Set entry price for stop-loss calculation

            # reward = 0  # No immediate reward

        elif action == 2:  # Sell
            # Sell all shares held
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.entry_price = 0

            # reward = 0  # No immediate reward

        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Calculate profit as reward
        profit = self.net_worth - INITIAL_BALANCE
        # reward = profit
        reward = self.net_worth - self.prev_net_worth

        transaction_cost = self.calculate_transaction_cost(action)
        reward -= transaction_cost

        # (Optional) Apply risk penalty
        # risk_penalty = risk_factor * self.calculate_risk()
        # reward -= risk_penalty

        done = self.current_step >= self.total_steps - 1

        # Get next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_BALANCE
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.current_step = 0
        return self._next_observation()
