# Reinforcement Learning Stock Trading Agent

This project implements a reinforcement learning (RL) agent using PyTorch to develop a stock trading strategy. The agent leverages deep learning techniques, specifically an LSTM-based Deep Q-Network (DQN), to make trading decisions based on historical stock data and technical indicators.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The goal of this project is to train an RL agent that can make profitable trading decisions by learning from historical stock data. The agent uses a window of past observations and technical indicators to predict the best action to take at each time step: hold, buy, or sell.

Key components:

- **Data Preparation**: Loading and preprocessing historical stock data, computing technical indicators, and normalizing the data.
- **Environment**: A custom OpenAI Gym environment that simulates stock trading, including risk management through dynamic stop-losses based on the Average True Range (ATR).
- **Agent**: An LSTM-based DQN agent capable of handling sequential data and making informed trading decisions.
- **Training**: Training the agent using experience replay and the epsilon-greedy strategy.
- **Evaluation**: Assessing the agent's performance and visualizing results.

## Features

- Uses a window of past observations to capture temporal dependencies.
- Incorporates technical indicators such as Moving Averages, Bollinger Bands, and ATR.
- Implements dynamic stop-losses based on market volatility.
- Normalizes data using percentage change to improve training stability.
- Employs an LSTM-based neural network to handle sequential data.
- Includes risk management strategies to limit potential losses.
- Saves a plot of the agent's net worth over time for performance evaluation.

## Installation

### Prerequisites

- Python 3.7 or higher
- Recommended libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `torch` (PyTorch)
  - `gym`

### Install Dependencies

You can install the required Python libraries using `pip`:

```bash
pip install numpy pandas matplotlib torch gym
```

## Usage

### 1. Prepare the Data

- Place your historical stock data in a CSV file named `stock_data.csv` in the same directory as the script.
- The CSV file should contain at least the following columns:

  - `Date`
  - `Open`
  - `High`
  - `Low`
  - `Close`
  - `Volume`

- Ensure the data is clean and free of errors.

### 2. Run the Script

- Save the provided code into a Python script file, e.g., `trading_rl_agent.py`.
- Execute the script:

  ```bash
  python trading_rl_agent.py
  ```

### 3. View the Results

- After training and evaluation, a plot of the agent's net worth over time will be saved as `net_worth_over_time.png`.
- Open this image to assess the agent's performance.

## Project Structure

```
trading_rl_agent.py      # Main script containing the code
dqn_agent.py             # RL agent implementation
stock_data.csv           # Historical stock data (provided by the user)
net_worth_over_time.png  # Output plot after evaluation
README.md                # Project documentation
```

## Technical Details

### Data Preparation

- **Technical Indicators**:
  - **Moving Averages**:
    - Short-term MA: 10-day Simple Moving Average (SMA)
    - Medium-term MA: 20-day Exponential Moving Average (EMA)
    - Long-term MA: 50-day SMA
  - **Bollinger Bands**:
    - Middle Band: 20-day SMA
    - Upper and Lower Bands: 2 standard deviations from the Middle Band
  - **Average True Range (ATR)**:
    - Measures market volatility over 14 days

- **Normalization**:
  - Percentage change normalization is applied to capture day-over-day changes.
  - Additional state variables (balance, shares held, net worth) are normalized as a percentage of the initial balance.

### Environment (`TradingEnv`)

- **Observation Space**:
  - A sequence of past observations (`n_steps`), each containing:
    - Normalized technical indicators and price data
    - Normalized state variables
- **Action Space**:
  - Discrete actions: `0` (Hold), `1` (Buy), `2` (Sell)
- **Reward Function**:
  - Calculated as the step-wise change in net worth
- **Risk Management**:
  - Dynamic stop-loss based on ATR to limit losses
  - Position sizing to ensure no more than 5% of net worth is risked per trade

### Agent (`DQNAgent`)

- **Architecture**:
  - LSTM layer to process sequential data
  - Fully connected layers to output Q-values for each action
- **Hyperparameters**:
  - Hidden size: 64
  - Learning rate: 0.001
  - Exploration rate (`epsilon`): Starts at 1.0 and decays to 0.01
- **Training**:
  - Experience replay with a memory buffer of size 2000
  - Trained over multiple episodes with minibatches of size 32

### Training and Evaluation

- **Training Loop**:
  - Agent interacts with the environment, selects actions using an epsilon-greedy strategy
  - Experiences are stored and sampled to train the agent using the DQN algorithm
- **Evaluation**:
  - Agent is evaluated without exploration to assess performance
  - Net worth over time is plotted and saved

## Results

- The agent's performance can be visualized in the `net_worth_over_time.png` plot.
- Final net worth and total reward from the evaluation are printed in the console.
- Results will vary based on the data used and the hyperparameters set.

## Disclaimer

This project is intended for educational purposes only. Trading in financial markets involves significant risk. The strategies and models presented do not guarantee profitable trading results. Past performance is not indicative of future results. Always conduct thorough research and consider consulting a professional financial advisor before making investment decisions.

## License

This project is licensed under the GPL v3. License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **OpenAI Gym**: For providing a framework to build custom reinforcement learning environments.
- **PyTorch**: For offering a flexible deep learning library.
- **Various online resources and tutorials** that have contributed to the development of this project.

---

Feel free to modify and enhance this project according to your needs. Contributions and feedback are welcome!