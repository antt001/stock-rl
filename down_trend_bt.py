import yfinance as yf
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import bs4 as bs
import requests

# Define the date range for the backtest
start_date = '2020-01-01'
end_date = '2023-12-31' # Use a date in the past for proper backtesting

# Fetch S&P 500 tickers
print("Fetching S&P 500 tickers list...")
resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'id': 'constituents'})

tickers = []

for row in table.findAll('tr')[1:]:
    td_tags = row.findAll('td')
    if len(td_tags) < 1:
        continue
    ticker = td_tags[0].text.strip()
    tickers.append(ticker)

print(f"Found {len(tickers)} tickers.")

for ticker in tickers:
    print(f"\n--- Backtesting {ticker} ---")
    # Download data
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError("No data downloaded.")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        continue
# --- Configuration ---
# Define the ticker symbol for the stock
ticker = 'AAPL'

# Initial capital for the backtest
initial_balance = 10000

# Transaction fee per trade leg (e.g., 0.1% buy + 0.1% sell = 0.2% total per round trip)
# Represented as a decimal (0.1% = 0.001)
transaction_fee_percent = 0.001

# Bollinger Band parameters
bb_window = 20
bb_std_dev = 2

# Profit target percentage (e.g., 5% gain)
profit_target_percent = 0.05

# --- Data Loading ---
print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
try:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError("No data downloaded.")
    print("Data downloaded successfully.")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit() # Exit if data download fails

# Ensure data is sorted by date
df.sort_index(inplace=True)

# --- Technical Analysis ---
print("Calculating technical indicators (Bollinger Bands)...")
# Calculate Bollinger Bands using the 'talib' library
# talib.BBANDS returns upper, middle, lower bands
df['BBU'], df['BBM'], df['BBL'] = talib.BBANDS(df['Close'].values.reshape(-1), timeperiod=bb_window, nbdevup=bb_std_dev, nbdevdn=bb_std_dev, matype=0) # matype=0 for SMA

# Clean up NaN values created by the lookback window
df.dropna(inplace=True)

# --- Signal Generation ---
print("Generating trading signals based on strategy rules...")

# Define strategy conditions
# 1. Green Candle: Close > Open
is_green_candle = df['Close'] > df['Open']

# 2. Body below Lower Bollinger Band
# The body spans from min(Open, Close) to max(Open, Close)
df['Body_Max'] = df[['Open', 'Close']].max(axis=1)
df['Body_Min'] = df[['Open', 'Close']].min(axis=1)
# Condition: The highest point of the body must be below the lower band
is_body_below_bb = df['Body_Max'] < df['BBL']

# 3. Body does not touch previous day's body (current body is below previous body)
# Need previous day's body max and min
df['Prev_Body_Max'] = df['Body_Max'].shift(1)
df['Prev_Body_Min'] = df['Body_Min'].shift(1)
# Condition: Current body max must be below previous day's body min
is_body_not_touching_prev = df['Body_Max'] < df['Prev_Body_Min']

# Combine conditions to find raw buy signals (signal occurs on day i)
# Requires data from day i and day i-1, so handle NaNs from shifting
# Combine conditions to find raw buy signals
condition = is_green_candle.values.flatten() & is_body_below_bb.values & is_body_not_touching_prev.values
# Fill NaN resulting from the shift(1) in is_body_not_touching_prev with False
# condition = condition.fillna(False)

# Assign signals using np.where based on the condition
# Create a Series with the correct index first to ensure alignment
signal_values = pd.Series(np.where(condition, 1, 0), index=df.index)
df['Signal'] = signal_values

# Store the Low of the signal candle, needed for stop loss
df['Signal_Low'] = np.nan
# Assign 'Low' value where Signal is 1, using .loc for clarity
df.loc[df['Signal'] == 1, 'Signal_Low'] = df.loc[df['Signal'] == 1, 'Low'].values

# Calculate the potential entry price for the *next* day (middle of the signal candle)
mask = df['Signal'] == 1
df.loc[mask, 'Potential_Entry_Price'] = ((
    df.loc[mask, 'High'] +
    df.loc[mask, 'Low']
) / 2).values

# Shift the potential entry price and signal low forward by one day, as the entry is attempted *the day after* the signal
df['Entry_Price_Limit'] = df['Potential_Entry_Price'].shift(1)
df['Stop_Loss_Level'] = df['Signal_Low'].shift(1)

# Clean up temporary columns
df.drop(columns=['Body_Max', 'Body_Min', 'Prev_Body_Max', 'Prev_Body_Min', 'Potential_Entry_Price', 'Signal_Low'], inplace=True)

# --- Backtesting Logic ---
print("Running backtesting simulation...")

balance = initial_balance
position = False
buy_price = 0
entry_date = None
stop_loss_price = 0
profit_target_price = 0

# Store balance history for plotting
balance_history = pd.Series(initial_balance, index=df.index)

# Store trade points for plotting
buy_points = []
sell_points = []
trade_results = [] # To log details of each completed trade

# Iterate through the data starting from the first possible entry day (after signal check and BB window)
# We iterate from the day *after* the first possible signal day
# Find the first index where shifted data (entry/stop loss levels) is available
valid_index_for_signal = df[['Entry_Price_Limit', 'Stop_Loss_Level']].dropna().index
if not valid_index_for_signal.empty:
    start_idx = df.index.get_loc(valid_index_for_signal[0])
else:
    print("Error: No valid data points found after calculations and shifts.")
    exit() # Or handle appropriately

# Adjust start_idx: The loop should start checking from the day identified by start_idx
# The original logic added +1, implying it wanted to start the day *after* the first valid signal day.
# Let's keep the original intent: start iterating from the day *after* the first day where entry/stop are defined.
start_idx += 1
# Original start_idx calculation removed (handled above)

for i in range(start_idx, len(df)):
    current_date = df.index[i]
    previous_date = df.index[i-1] # Date of the potential signal

    # Always update balance history at the start of the day
    balance_history.iloc[i] = balance

    # --- Check for Entry ---
    # We are not in a position AND a signal occurred on the previous day AND we have a potential entry price
    if not position and df['Signal'].iloc[i-1] == 1 and not pd.isna(df['Entry_Price_Limit'].iloc[i]):
        # Check if today's low price hit the limit order set from yesterday's signal
        if df['Low'].values.reshape(-1)[i] <= df['Entry_Price_Limit'].iloc[i]:
            # Trade executed!
            position = True
            buy_price = df['Entry_Price_Limit'].iloc[i] # Filled at the limit price
            entry_date = current_date
            stop_loss_price = df['Stop_Loss_Level'].iloc[i] # SL based on low of signal candle
            profit_target_price = buy_price * (1 + profit_target_percent)

            # Record buy point for plotting
            buy_points.append({'date': current_date, 'price': buy_price})

            # print(f"BUY on {current_date.date()} at {buy_price:.2f} (Signal on {previous_date.date()}) SL: {stop_loss_price:.2f}, PT: {profit_target_price:.2f}")


    # --- Check for Exit (if in a position) ---
    if position:
        exit_price = None
        exit_type = None

        # Check Profit Target hit during the day
        if df['High'].values.reshape(-1)[i] >= profit_target_price:
            exit_price = profit_target_price
            exit_type = 'Target'
            # print(f"SELL Target: {current_date.date()} at {exit_price:.2f}")

        # Check Stop Loss hit (based on TODAY's CLOSE vs. Signal Candle's Low)
        # Only check if target wasn't already hit
        elif df['Close'].values.reshape(-1)[i] <= stop_loss_price:
             exit_price = df['Close'].values.reshape(-1)[i] # Exit at the closing price
             exit_type = 'Stop'
             # print(f"SELL Stop: {current_date.date()} at {exit_price:.2f}")

        # End of Day Exit (Mandatory close for single-day strategy)
        # This logic triggers if neither target nor stop was hit *by the close*
        # In a real single-day trade, you'd place a market order near the close if not exited
        # Here, we just use the closing price as the exit price
        if exit_price is None:
             exit_price = df['Close'].values.reshape(-1)[i]
             exit_type = 'EOD'
             # print(f"SELL EOD: {current_date.date()} at {exit_price:.2f}")

        # If any exit condition was met (which one always will be by the end of the day)
        if exit_price is not None:
            gross_profit = exit_price - buy_price
            # Calculate fee on both buy and sell legs based on their respective prices
            # For simplicity here, let's calculate the fee on the average of buy/sell price
            # Or, just apply 2*fee_percent to the gross percentage return as an approximation
            gross_profit_pct = gross_profit / buy_price
            net_profit_pct = gross_profit_pct - (2 * transaction_fee_percent) # Fee on buy and sell leg

            # Apply the percentage P/L to the balance. This is a simplification,
            # real backtesting manages position size (number of shares) more explicitly.
            # This assumes the 'unit' of trade scales with the balance.
            profit_amount = balance * net_profit_pct
            balance += profit_amount

            # Record sell point for plotting
            sell_points.append({'date': current_date, 'price': exit_price, 'type': exit_type})

            # Record trade result
            trade_results.append({
                'entry_date': entry_date,
                'buy_price': buy_price,
                'exit_date': current_date,
                'sell_price': exit_price,
                'exit_type': exit_type,
                'gross_p_l_pct': gross_profit_pct * 100,
                'net_p_l_pct': net_profit_pct * 100,
                'balance_after': balance
            })

            # Reset position state
            position = False
            buy_price = 0
            entry_date = None
            stop_loss_price = 0
            profit_target_price = 0

# --- Results ---
print("\n--- Backtest Summary ---")
final_balance = balance
total_return_pct = (final_balance - initial_balance) / initial_balance * 100
num_trades = len([t for t in trade_results])

print(f"Ticker: {ticker}")
print(f"Date Range: {start_date} to {end_date}")
print(f"Initial Balance: ${initial_balance:.2f}")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Total Return: {total_return_pct:.2f}%")
print(f"Transaction Fee per Leg: {transaction_fee_percent:.3f}%")
print(f"Number of Completed Trades: {num_trades}")

if num_trades > 0:
    avg_net_profit_pct = np.mean([t['net_p_l_pct'] for t in trade_results])
    winning_trades = [t for t in trade_results if t['net_p_l_pct'] > 0]
    losing_trades = [t for t in trade_results if t['net_p_l_pct'] <= 0]
    win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
    avg_win_pct = np.mean([t['net_p_l_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss_pct = np.mean([t['net_p_l_pct'] for t in losing_trades]) if losing_trades else 0
    max_win_pct = np.max([t['net_p_l_pct'] for t in trade_results]) if trade_results else 0
    max_loss_pct = np.min([t['net_p_l_pct'] for t in trade_results]) if trade_results else 0


    print(f"Average Net Profit per Trade: {avg_net_profit_pct:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Winning Trade: {avg_win_pct:.2f}%")
    print(f"Average Losing Trade: {avg_loss_pct:.2f}%")
    print(f"Max Winning Trade: {max_win_pct:.2f}%")
    print(f"Max Losing Trade: {max_loss_pct:.2f}%")


print("\n--- Trade Log ---")
if num_trades > 0:
    for trade in trade_results:
         print(f"BUY: {trade['entry_date'].date()} at {trade['buy_price']:.2f} | SELL: {trade['exit_date'].date()} at {trade['sell_price']:.2f} ({trade['exit_type']}) | Net P/L: {trade['net_p_l_pct']:.2f}% | Balance: ${trade['balance_after']:.2f}")
else:
    print("No trades were executed during the backtest period based on the strategy signals.")


# --- Plotting ---
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# --- Price Chart with Trades and Indicators ---
ax1 = axes[0]
ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
ax1.plot(df.index, df['BBL'], label=f'BB Lower ({bb_window},{bb_std_dev})', color='red', linestyle='--', alpha=0.7)

# Plot Buy points
buy_dates = [p['date'] for p in buy_points]
buy_prices = [p['price'] for p in buy_points]
ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy (Executed)', zorder=5, alpha=0.8)

# Plot Sell points with different markers for exit types
sell_dates = [p['date'] for p in sell_points]
sell_prices = [p['price'] for p in sell_points]
sell_types = [p['type'] for p in sell_points]

# Separate sell points by type for distinct markers
target_sells = [{'date': d, 'price': p} for d, p, t in zip(sell_dates, sell_prices, sell_types) if t == 'Target']
stop_sells = [{'date': d, 'price': p} for d, p, t in zip(sell_dates, sell_prices, sell_types) if t == 'Stop']
eod_sells = [{'date': d, 'price': p} for d, p, t in zip(sell_dates, sell_prices, sell_types) if t == 'EOD']


if target_sells:
    ax1.scatter([p['date'] for p in target_sells], [p['price'] for p in target_sells], marker='v', color='cyan', s=100, label='Sell (Target)', zorder=5, alpha=0.8)
if stop_sells:
    ax1.scatter([p['date'] for p in stop_sells], [p['price'] for p in stop_sells], marker='X', color='orange', s=100, label='Sell (Stop)', zorder=5, alpha=0.8)
if eod_sells:
     ax1.scatter([p['date'] for p in eod_sells], [p['price'] for p in eod_sells], marker='s', color='red', s=100, label='Sell (EOD)', zorder=5, alpha=0.8)


ax1.set_ylabel('Price ($)')
ax1.set_title(f'{ticker} Price Chart with Trades and BB Lower Band ({start_date} to {end_date})')
ax1.legend()
ax1.grid(True)

# --- Balance Chart ---
ax2 = axes[1]
ax2.plot(balance_history.index, balance_history, label='Account Balance', color='purple')
ax2.set_ylabel('Balance ($)')
ax2.set_title('Account Balance Over Time')
ax2.legend()
ax2.grid(True)

# Improve date formatting on x-axis
fig.autofmt_xdate()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_minor_locator(mdates.MonthLocator())


plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()

print("Backtesting complete.")