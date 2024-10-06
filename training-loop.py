import random
from collections import deque

from trading_env import TradingEnv
from dqn_agent import DQNAgent
from evaluation import evaluate_agent
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from load_data import load_data

import matplotlib.pyplot as plt

# Before initializing the environment
n_steps = 10  # Adjust as needed

current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
def train_dqn(env, agent, episodes=50, batch_size=32, gamma=0.99,
              epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
              save_path=f'best_model_{current_datetime}.pth'):
    memory = deque(maxlen=2000)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    best_net_worth = env.initial_balance  # Initialize best net worth

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
        
        # At the end of the episode, check if net worth improved
        final_net_worth = env.net_worth
        if final_net_worth > best_net_worth:
            best_net_worth = final_net_worth
            # Save the model
            torch.save(agent.state_dict(), save_path)
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, "
                  f"New Best Net Worth: ${best_net_worth:.2f}, Model Saved.")
        else:
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, "
                  f"Net Worth: ${final_net_worth:.2f}")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # Save the model
    torch.save(agent.state_dict(), f'last_model_{current_datetime}.pth')
    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, "
          f"final Net Worth: ${final_net_worth:.2f}, Model Saved.")

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

df = load_data('AAPL', start='2020-01-01', end='2023-12-31')

# Set the window size for past observations
n_steps = 10  # Adjust as needed

# Initialize the environment
env = TradingEnv(df, n_steps=n_steps, fee_structure='per_share')

# Get the input size from the environment
input_size = env.observation_space.shape[1]
action_size = env.action_space.n

# Initialize the agent
agent = DQNAgent(input_size, action_size)

# Define the path to save the best model
model_save_path = f'best_model_{current_datetime}.pth'

# Train the agent
train_dqn(env, agent, episodes=100, batch_size=32, save_path=model_save_path)

# Evaluate the agent
evaluate_agent(env, agent, load_path=model_save_path)
