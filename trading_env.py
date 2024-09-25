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
    def __init__(self, df, n_steps=10, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.n_steps = n_steps
        self.initial_balance = initial_balance
        self.total_steps = len(df) - 1
        self.current_step = self.n_steps  # Start from n_steps to have enough data

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = Discrete(3)

        # Initialize state variables
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0
        self.stop_loss_price = None
        self.prev_net_worth = self.net_worth

        # Observation space dimensions
        self.num_features = 11  # Number of features per time step
        self.additional_vars = 3  # balance, shares held, net worth
        obs_shape = (self.n_steps, self.num_features + self.additional_vars)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def _next_observation(self):
        # Get the data points for the last n_steps days
        end = self.current_step + 1
        start = end - self.n_steps

        history = self.df.iloc[start:end]

        # Extract features
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'MA_Short', 'MA_Medium', 'MA_Long',
                    'BB_Upper', 'BB_Lower', 'ATR']

        # Calculate percentage change
        obs = history[features].pct_change().fillna(0).values

        # Include additional state variables
        additional_vars = np.array([self.balance, self.shares_held, self.net_worth])
        # Normalize additional_vars as percentage of initial balance
        additional_vars_normalized = (additional_vars - self.initial_balance) / self.initial_balance
        additional_vars_normalized = np.tile(additional_vars_normalized, (obs.shape[0], 1))
        obs = np.concatenate((obs, additional_vars_normalized), axis=1)

        return obs.astype(np.float32)

    def step(self, action):
        # Save previous net worth
        self.prev_net_worth = self.net_worth

        # Get current price and ATR
        current_price = self.df.loc[self.current_step, 'Close']
        current_atr = self.df.loc[self.current_step, 'ATR']

        # Execute action
        if action == 1:  # Buy
            # Determine position size based on risk management
            max_loss = self.net_worth * 0.05  # Risk no more than 5% of net worth
            atr_multiplier = 1  # Adjust as needed
            stop_loss_distance = current_atr * atr_multiplier
            position_size = max_loss / stop_loss_distance

            # Ensure we don't buy more than we can afford
            shares_to_buy = min(
                position_size, self.balance // current_price
            )
            self.balance -= shares_to_buy * current_price
            self.shares_held += shares_to_buy
            self.entry_price = current_price  # Set entry price for stop-loss calculation

            # Calculate stop-loss price
            self.stop_loss_price = self.entry_price - (current_atr * atr_multiplier)

        elif action == 2:  # Sell
            # Sell all shares held
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.entry_price = 0
            self.stop_loss_price = None

        else:  # Hold
            pass

        # Check if stop-loss is hit
        if self.shares_held > 0 and current_price <= self.stop_loss_price:
            # Sell all shares held
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.entry_price = 0
            self.stop_loss_price = None
            print(f"Stop-loss triggered at price {current_price:.2f}")

        # Update net worth after action
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Calculate reward as step-wise profit/loss
        reward = self.net_worth - self.prev_net_worth

        # Update previous net worth for next step
        self.prev_net_worth = self.net_worth

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= self.total_steps

        # Get next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0
        self.stop_loss_price = None
        self.prev_net_worth = self.net_worth
        self.current_step = self.n_steps  # Start from n_steps to have enough data
        return self._next_observation()
