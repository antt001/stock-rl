
import matplotlib.pyplot as plt
import torch

from trading_env import TradingEnv
from dqn_agent import DQNAgent
from datetime import datetime
from load_data import load_data

def evaluate_agent(env, agent, load_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    # Load the saved model
    agent.load_state_dict(torch.load(load_path, map_location=device))
    agent.eval()  # Set the agent to evaluation mode

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    total_reward = 0
    net_worths = []

    prices = []  # To store the prices over time
    adx_values = []
    rsi_values = []
    actions = []  # To store the actions taken
    trade_positions = []  # To store entry and exit points
    timestamps = []  # To store time steps

    prev_shares_held = env.shares_held  # Track shares held from the previous step

    for t in range(env.total_steps - env.n_steps):
        with torch.no_grad():
            q_values = agent(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        total_reward += reward
        net_worths.append(env.net_worth)

         # Get current price
        current_price = env.df.loc[env.current_step - 1, 'Close']
        prices.append(current_price)

        adx = env.df.loc[env.current_step - 1, 'ADX']
        adx_values.append(adx)

        # During the evaluation loop
        current_rsi = env.df.loc[env.current_step - 1, 'RSI']
        rsi_values.append(current_rsi)

        timestamps.append(t)

        # Record trade actions
        if action == 1 and prev_shares_held == 0 and env.shares_held > 0:
            # Buy action
            trade_positions.append({
                'time': t,
                'price': current_price,
                'type': 'Buy'
            })
        elif prev_shares_held > 0 and env.shares_held == 0:
            # Sell or Stop-Loss action
            if action == 2:
                trade_type = 'Sell'
            else:
                trade_type = 'Stop-Loss'
            # Sell action
            trade_positions.append({
                'time': t,
                'price': current_price,
                'type': trade_type
            })
        
        # Update previous shares held
        prev_shares_held = env.shares_held
        if done:
            break

    # Plotting net worth over time
    plt.figure(figsize=(12, 6))
    plt.plot(net_worths)
    plt.title('Agent Net Worth Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Worth ($)')
    plt.savefig('net_worth_over_time.png')
    plt.close()

    # Plotting price chart with entry and exit points and ADX
    fig, ax1 = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)
    # Plot the price on the first subplot
    ax1[0].plot(timestamps, prices, label='Price')

    # Extract trade positions
    buy_times = [trade['time'] for trade in trade_positions if trade['type'] == 'Buy']
    buy_prices = [trade['price'] for trade in trade_positions if trade['type'] == 'Buy']
    sell_times = [trade['time'] for trade in trade_positions if trade['type'] == 'Sell']
    sell_prices = [trade['price'] for trade in trade_positions if trade['type'] == 'Sell']
    stoploss_times = [trade['time'] for trade in trade_positions if trade['type'] == 'Stop-Loss']
    stoploss_prices = [trade['price'] for trade in trade_positions if trade['type'] == 'Stop-Loss']

    # Plot trade markers
    ax1[0].scatter(buy_times, buy_prices, marker='^', color='g', label='Buy', s=100)
    ax1[0].scatter(sell_times, sell_prices, marker='v', color='r', label='Sell', s=100)
    ax1[0].scatter(stoploss_times, stoploss_prices, marker='x', color='k', label='Stop-Loss[Trailing]', s=100)

    ax1[0].set_title('Price Chart with Entry and Exit Points')
    ax1[0].set_xlabel('Time Steps')
    ax1[0].set_ylabel('Price')
    ax1[0].legend()
    ax1[0].grid(True)

    # Plot ADX on the second subplot
    # ax1[1].plot(timestamps, adx_values, label='ADX', color='b')
    # ax1[1].axhline(y=25, color='r', linestyle='--', label='ADX Threshold (25)')
    # ax1[1].set_title('Average Directional Index (ADX)')
    # ax1[1].set_xlabel('Time Steps')
    # ax1[1].set_ylabel('ADX Value')
    # ax1[1].legend()
    # ax1[1].grid(True)

    # Plot RSI on the third subplot
    ax1[1].plot(timestamps, rsi_values, label='RSI', color='purple')
    ax1[1].axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    ax1[1].axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    ax1[1].set_title('Relative Strength Index (RSI)')
    ax1[1].set_xlabel('Time Steps')
    ax1[1].set_ylabel('RSI Value')
    ax1[1].legend()
    ax1[1].grid(True)

    plt.tight_layout()
    plt.savefig('price_chart_with_trades_and_adx.png')
    plt.close()

    print(f"Final Net Worth: ${env.net_worth:.2f}")
    print(f"Total Reward from Evaluation: {total_reward:.2f}")

if __name__ == "__main__":

    df = load_data('AAPL', start='2024-01-01', end='2024-12-31')
    # Before initializing the environment
    n_steps = 10  # Adjust as needed

    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    # model_save_path = f'best_model_{current_datetime}.pth'
    model_save_path = 'best_model_20241003153923.pth'
    env = TradingEnv(df, n_steps=n_steps, fee_structure='per_share')

    # Get the input size from the environment
    input_size = env.observation_space.shape[1]
    action_size = env.action_space.n

    # Initialize the agent
    agent = DQNAgent(input_size, action_size)

    # Evaluate the agent
    evaluate_agent(env, agent, load_path=model_save_path)