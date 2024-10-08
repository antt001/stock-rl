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
    def __init__(self, df, n_steps=10, initial_balance=10000, fee_structure='percentage'):
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
        self.fee_structure = fee_structure  # 'percentage' or 'per_share'
        # Trailing Stop-Loss Variables
        self.trailing_stop_loss = True  # Enable or disable trailing stop-loss
        self.trailing_stop_distance = 0.02  # 1% trailing stop-loss
        self.highest_price = 0  # Highest price since entering the position

        # Extract features
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'MA_Short', 'MA_Medium', 'MA_Long',
                    'BB_Upper', 'BB_Lower', 'ATR',
                    'MA_Difference', 
                    # 'MA_Crossover',
                    # 'ADX', 
                    # 'MACD', 
                    # 'MACD_Signal', 'RSI'
                    ]

        # Observation space dimensions
        self.num_features = len(self.features) # Number of features per time step
        self.additional_vars = 3  # balance, shares held, net worth
        obs_shape = (self.n_steps, self.num_features + self.additional_vars)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def calculate_transaction_cost_percentage(self, trade_value):
        fee_rate = 0.001  # 0.1% fee
        transaction_cost = fee_rate * trade_value
        return transaction_cost

    def calculate_transaction_cost_per_share(self, shares_traded):
        cost_per_share = 0.01  # $0.01 per share
        transaction_cost = shares_traded * cost_per_share
        minimum_fee = 6.5
        if transaction_cost < minimum_fee:
            transaction_cost = minimum_fee
        return transaction_cost

    def _next_observation(self):
        # Get the data points for the last n_steps days
        end = self.current_step + 1
        start = end - self.n_steps

        history = self.df.iloc[start:end]

        # Calculate percentage change
        obs = history[self.features].pct_change().fillna(0).values

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

        # Update the highest price and trailing stop-loss
        if self.shares_held > 0:
            if current_price > self.highest_price:
                self.highest_price = current_price
                # Update trailing stop-loss price
                if self.trailing_stop_loss:
                    self.stop_loss_price = self.highest_price * (1 - self.trailing_stop_distance)
        else:
            self.highest_price = 0  # Reset when not in position

         # Determine transaction cost based on fee structure
        def get_transaction_cost(trade_value, shares_traded):
            if self.fee_structure == 'percentage':
                return self.calculate_transaction_cost_percentage(trade_value)
            elif self.fee_structure == 'per_share':
                return self.calculate_transaction_cost_per_share(shares_traded)
            else:
                return 0

        # Execute action
        if action == 1:  # Buy
            # Determine position size based on risk management
            max_loss = self.net_worth * 0.02  # Risk no more than 2% of net worth
            atr_multiplier = 1  # Adjust as needed
            stop_loss_distance = current_atr * atr_multiplier
            position_size = max_loss / stop_loss_distance

            # Ensure we don't buy more than we can afford
            shares_to_buy = min(
                position_size, self.balance // current_price
            )

            if shares_to_buy > 0:
                # Calculate trade value
                trade_value = shares_to_buy * current_price

                # Calculate transaction cost
                transaction_cost = get_transaction_cost(trade_value, shares_to_buy)

                # Update balance and shares held
                total_cost = trade_value + transaction_cost

                self.balance -= total_cost
                self.shares_held += shares_to_buy
                self.entry_price = current_price  # Set entry price for stop-loss calculation

                # Initialize highest price and stop-loss price
                self.highest_price = current_price
                if self.trailing_stop_loss:
                    self.stop_loss_price = self.highest_price * (1 - self.trailing_stop_distance)
                else:
                    self.stop_loss_price = self.entry_price - (current_atr * atr_multiplier)
            else:
                # Not enough balance to buy
                pass

        elif action == 2:  # Sell
            # Sell all shares held
            # Calculate trade value
            trade_value = self.shares_held * current_price
            # Calculate transaction cost
            transaction_cost = get_transaction_cost(trade_value, self.shares_held)
            # Update balance and shares held
            total_proceeds = trade_value - transaction_cost
            self.balance += total_proceeds
            self.shares_held = 0
            self.entry_price = 0
            # Reset highest price and stop-loss price
            self.highest_price = 0
            self.stop_loss_price = None

        else:  # Hold
            pass

        # Check if stop-loss is hit
        if self.shares_held > 0 and current_price <= self.stop_loss_price:
            # Sell all shares held
             # Calculate trade value
            trade_value = self.shares_held * current_price

            # Calculate transaction cost
            transaction_cost = get_transaction_cost(trade_value, self.shares_held)

            # Update balance and shares held
            total_proceeds = trade_value - transaction_cost
            self.balance += total_proceeds
            self.shares_held = 0
            self.entry_price = 0
            self.stop_loss_price = None
            self.highest_price = 0
            # print(f"Stop-loss triggered at price {current_price:.2f}")

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
