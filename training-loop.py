import pandas as pd
from tqdm import tqdm
from trading_env import TradingEnv
from dqn_agent import DQNAgent

# Load and preprocess your data
df = pd.read_csv('2020_1h_ETHUSDT_binance.csv')
# Preprocess your data here (e.g., calculate technical indicators, normalize)
print(df.head())
print(df.iloc[0][4], df.columns)
# Create environment and agent
env = TradingEnv(df)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 1000
batch_size = 32

# Training loop
for episode in tqdm(range(episodes)):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    if episode % 10 == 0:
        print(f"Episode: {episode}, Epsilon: {agent.epsilon:.2f}, Final Balance: {env.balance:.2f}")

# Save the trained model
torch.save(agent.model.state_dict(), 'trading_model.pth')
