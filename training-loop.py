import random
from collections import deque
import yfinance as yf
from trading_env import TradingEnv
from dqn_agent import DQNAgent
import torch
import torch.nn as nn
import torch.optim as optim

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Before initializing the environment
n_steps = 10  # Adjust as needed

def train_dqn(env, agent, episodes=50, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    memory = deque(maxlen=2000)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        for t in range(env.total_steps):
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = agent(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
                break

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                train_minibatch(agent, optimizer, criterion, minibatch, gamma)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

def train_minibatch(agent, optimizer, criterion, minibatch, gamma):
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.stack(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.stack(next_states)
    dones = torch.FloatTensor(dones)

    # Current Q values
    current_q_values = agent(states).gather(1, actions)

    # Next Q values
    with torch.no_grad():
        max_next_q_values = agent(next_states).max(1)[0]

    # Target Q values
    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    # Compute loss
    loss = criterion(current_q_values.squeeze(), target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate_agent(env, agent):
    state = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    net_worths = []
    balances = []
    positions = []

    for t in range(env.total_steps):
        with torch.no_grad():
            q_values = agent(state)
            action = torch.argmax(q_values).item()
            print(f"Action: {action}")
            print(f"Q-Values: {q_values}")

        next_state, reward, done, _ = env.step(action)
        state = torch.FloatTensor(next_state)
        total_reward += reward
        net_worths.append(env.net_worth)
        balances.append(env.balance)
        positions.append(env.shares_held)

        if done:
            break

    # Plotting net worth over time
    plt.figure(figsize=(12, 6))
    plt.plot(net_worths)
    plt.title('Agent Net Worth Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Worth ($)')
    plt.show()

    print(f"Final Net Worth: ${env.net_worth:.2f}")
    print(f"Total Reward from Evaluation: {total_reward:.2f}")

# Training loop
# load data from yfinance
df = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
# Sort the data by date if not already sorted
df.sort_values('Date', inplace=True)

# Reset the index after sorting
df.reset_index(drop=True, inplace=True)

# (Optional) Feature engineering: Add technical indicators
# Compute Moving Averages
df['MA_Short'] = df['Close'].rolling(window=10).mean()
df['MA_Medium'] = df['Close'].ewm(span=20, adjust=False).mean()
df['MA_Long'] = df['Close'].rolling(window=50).mean()

# Compute Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
df['BB_STD'] = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (df['BB_STD'] * 2)
df['BB_Lower'] = df['BB_Middle'] - (df['BB_STD'] * 2)

# Compute True Range (TR)
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

# Compute ATR
df['ATR'] = df['TR'].rolling(window=14).mean()

# Clean up intermediate columns
df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)

# Handle NaN values after adding indicators
df.fillna(method='bfill', inplace=True)

# create environment
env = TradingEnv(df, n_steps=n_steps, scaler=scaler)
# Get the size of the state and action space
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the agent
agent = DQNAgent(state_size, action_size)

# Train the agent
train_dqn(env, agent, episodes=1000, batch_size=32)

# Evaluate the agent
evaluate_agent(env, agent)
