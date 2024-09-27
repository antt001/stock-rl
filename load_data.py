import yfinance as yf

def load_data(ticker, start, end):
    # load data from yfinance
    df = yf.download(ticker, start=start, end=end)
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

    return df
