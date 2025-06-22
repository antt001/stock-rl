import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data # Modified version
from trading_env import TradingEnv, INITIAL_BALANCE # Modified version, import INITIAL_BALANCE
from datetime import datetime

def run_rule_based_strategy(ticker='AAPL', start_date='2020-01-01', end_date='2023-12-31',
                            n_steps_env=10, initial_balance_env=INITIAL_BALANCE, fee_structure_env='per_share'):
    """
    Runs the rule-based trading strategy and plots results.
    """
    # Load Data
    df = load_data(ticker, start=start_date, end=end_date)
    if df.empty or len(df) <= n_steps_env:
        print(f"Not enough data for {ticker} in the given date range after loading and n_steps consideration.")
        return

    # Initialize Environment
    # scaler=None is used because this agent uses raw dataframe values for decisions,
    # not the scaled observation space directly.
    env = TradingEnv(df, scaler=None, n_steps=n_steps_env, 
                     initial_balance=initial_balance_env, fee_structure=fee_structure_env)

    # Trading Loop
    net_worths = []
    prices_history = [] # Store close prices for the main price chart
    trade_positions = [] # To store entry and exit points {'time': t, 'price': price, 'type': 'Buy'/'Stop-Loss'}
    timestamps_plot = [] # For x-axis of plots

    downward_sequence_active = False # For pivot point logic

    env.reset() # Ensure environment is reset

    for t_step in range(env.total_steps - env.n_steps): # Loop for the number of possible steps
        current_df_idx = env.current_step # DataFrame index for current decision
        
        # Ensure current_df_idx and current_df_idx-1 are valid
        if current_df_idx == 0: # Should not happen if n_steps_env > 0 and loop starts correctly
            # This step is just before the first possible action, env.step() will advance it
            # For safety, if it's the very first data point, we can't get previous_rsi.
            # However, env.current_step starts at n_steps, so current_df_idx-1 is safe.
            pass

        # Get current indicator values from DataFrame
        current_ma_medium = df.loc[current_df_idx, 'MA_Medium'].values
        current_ma_long = df.loc[current_df_idx, 'MA_Long'].values
        current_close = df.loc[current_df_idx, 'Close'].values
        current_adx = df.loc[current_df_idx, 'ADX'].values
        current_plus_di = df.loc[current_df_idx, 'PLUS_DI'].values
        current_minus_di = df.loc[current_df_idx, 'MINUS_DI'].values
        previous_high = df.loc[current_df_idx - 1, 'High'].values
        current_volume = df.loc[current_df_idx, 'Volume'].values
        current_volume_ma = df.loc[current_df_idx, 'Volume_MA'].values

        # Apply Trading Rules
        is_uptrend_ma = current_ma_medium > current_ma_long and current_close > current_ma_medium
        is_uptrend_adx = current_adx > 25 and current_plus_di > current_minus_di
        general_uptrend_confirmed = True # is_uptrend_ma and is_uptrend_adx
        
        # Pivot point signal logic
        pivot_buy_signal = False
        if current_close > previous_high:
            if downward_sequence_active:
                pivot_buy_signal = True
            downward_sequence_active = False
        else:  # current_close <= previous_high
            downward_sequence_active = True
        
        volume_confirmed = current_volume > current_volume_ma
        can_buy = env.shares_held == 0

        action_to_take = 0  # Hold by default
        if general_uptrend_confirmed and pivot_buy_signal and volume_confirmed and can_buy:
            action_to_take = 1  # Buy

        prev_shares_held = env.shares_held
        
        # Step Environment
        # The environment uses df.loc[env.current_step] internally for price etc.
        obs_ignored, reward, done, info = env.step(action_to_take)

        # Record Data for Plotting
        net_worths.append(env.net_worth)
        prices_history.append(current_close) # Price at the time of decision
        timestamps_plot.append(t_step)

        # Record trade actions
        price_at_event = current_close # Price at which the decision/event occurred
        if action_to_take == 1 and prev_shares_held == 0 and env.shares_held > 0: # Buy
            trade_positions.append({
                'time': t_step,
                'price': price_at_event, # Or env.entry_price if more accurate after transaction
                'type': 'Buy'
            })
        elif prev_shares_held > 0 and env.shares_held == 0: # Position closed (must be Stop-Loss)
            trade_positions.append({
                'time': t_step,
                'price': price_at_event, # Price at which stop-loss might have been evaluated
                'type': 'Stop-Loss'
            })
        
        if done:
            print(f"Simulation ended early at step {t_step}.")
            break
    
    if not net_worths:
        print("No simulation steps were run, cannot generate plots or metrics.")
        return

    # Performance Metrics
    final_net_worth = env.net_worth
    # Buy and Hold Strategy Comparison
    buy_and_hold_start_price = df['Close'].iloc[0]
    buy_and_hold_end_price = df['Close'].iloc[-1]
    buy_and_hold_return = (buy_and_hold_end_price - buy_and_hold_start_price) / buy_and_hold_start_price
    buy_and_hold_net_worth = (initial_balance_env * (1 + buy_and_hold_return.values)[0])
    
    print(f"Final Net Worth (Rule-Based): ${final_net_worth:.2f}")
    print(f"Final Net Worth (Buy and Hold): ${buy_and_hold_net_worth:.2f}")

    net_worths_np = np.array(net_worths)
    if len(net_worths_np) > 1:
        returns = np.diff(net_worths_np) / net_worths_np[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        peak = np.maximum.accumulate(net_worths_np)
        drawdown = (net_worths_np - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
    else:
        print("Not enough data points to calculate Sharpe Ratio or Max Drawdown.")


    # Plotting Net Worth Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps_plot, net_worths, label='Rule-Based Strategy')
    # Plot Buy and Hold
    buy_and_hold_values = [initial_balance_env * (p / df['Close'].iloc[env.n_steps]) for p in prices_history]
    plt.plot(timestamps_plot, buy_and_hold_values, label='Buy and Hold', linestyle='--')
    plt.title(f'Strategy vs. Buy and Hold Net Worth Over Time ({ticker})')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Worth ($)')
    plt.legend()
    plt.savefig(f'rule_based_net_worth_{ticker}.png')
    plt.close()
    print(f"Net worth plot saved to rule_based_net_worth_{ticker}.png")

    # Plotting Price Chart with Entry and Exit Points
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps_plot, prices_history, label='Price', alpha=0.7)

    buy_times = [trade['time'] for trade in trade_positions if trade['type'] == 'Buy']
    buy_prices = [trade['price'] for trade in trade_positions if trade['type'] == 'Buy']
    stoploss_times = [trade['time'] for trade in trade_positions if trade['type'] == 'Stop-Loss']
    stoploss_prices = [trade['price'] for trade in trade_positions if trade['type'] == 'Stop-Loss']

    plt.scatter(buy_times, buy_prices, marker='^', color='g', label='Buy', s=100, alpha=1)
    plt.scatter(stoploss_times, stoploss_prices, marker='v', color='r', label='Stop-Loss', s=100, alpha=1)

    plt.title(f'Rule-Based Strategy Trades on Price Chart ({ticker})')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'rule_based_trades_{ticker}.png')
    plt.close()
    print(f"Trades plot saved to rule_based_trades_{ticker}.png")

if __name__ == "__main__":
    # Example usage:
    run_rule_based_strategy(ticker='AAPL', start_date='2022-01-01', end_date='2023-12-31')
    # run_rule_based_strategy(ticker='MSFT', start_date='2022-01-01', end_date='2023-12-31')