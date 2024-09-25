import random
from collections import deque
import yfinance as yf
from trading_env import TradingEnv
from dqn_agent import DQNAgent
from evaluation import evaluate_agent
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# Before initializing the environment
n_steps = 10  # Adjust as needed

def train_dqn(env, agent, episodes=50, batch_size=32, gamma=0.99,
              epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    memory = deque(maxlen=2000)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Shape: (1, n_steps, input_size)
        total_reward = 0

        for t in range(env.total_steps - env.n_steps):
            # Epsilon-greedy action selection
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = agent(state)
                    action = torch.argmax(q_values).item()

            # Take action and observe result
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            # Store experience in memory
            memory.append((state, action, reward, next_state_tensor, done))

            state = next_state_tensor
            total_reward += reward

            # Experience replay
            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                train_minibatch(agent, optimizer, criterion, minibatch, gamma, device)

            if done:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}")
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

def train_minibatch(agent, optimizer, criterion, minibatch, gamma, device):
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Stack states and next_states
    states = torch.cat(states)  # Shape: (batch_size, n_steps, input_size)
    next_states = torch.cat(next_states)

    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

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

# Reset index after adding indicators
df.reset_index(drop=True, inplace=True)

# Set the window size for past observations
n_steps = 10  # Adjust as needed

# Initialize the environment
env = TradingEnv(df, n_steps=n_steps)

# Get the input size from the environment
input_size = env.observation_space.shape[2]
action_size = env.action_space.n

# Initialize the agent
agent = DQNAgent(input_size, action_size)

# Train the agent
train_dqn(env, agent, episodes=50, batch_size=32)

# Evaluate the agent
evaluate_agent(env, agent)
