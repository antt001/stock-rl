import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model, env, episodes=1):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(model.device)
            with torch.no_grad():
                action = torch.argmax(model(state)).item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# Load test data
test_df = pd.read_csv('test_stock_data.csv')
test_env = TradingEnv(test_df)

# Load the trained model
trained_model = DQN(state_size, action_size).to(agent.device)
trained_model.load_state_dict(torch.load('trading_model.pth'))
trained_model.eval()

# Evaluate the model
average_reward = evaluate_model(trained_model, test_env)
print(f"Average reward on test data: {average_reward}")

# Backtest the strategy
state = test_env.reset()
done = False
portfolio_values = [test_env.initial_balance]

while not done:
    state = torch.FloatTensor(state).unsqueeze(0).to(trained_model.device)
    with torch.no_grad():
        action = torch.argmax(trained_model(state)).item()
    state, reward, done, _ = test_env.step(action)
    portfolio_values.append(test_env.balance + test_env.owned_shares * test_env.df.iloc[test_env.current_step]['close'])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(portfolio_values)
plt.title('Portfolio Value Over Time')
plt.xlabel('Trading Steps')
plt.ylabel('Portfolio Value')
plt.show()
