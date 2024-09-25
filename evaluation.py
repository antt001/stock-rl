
import matplotlib.pyplot as plt
import torch

def evaluate_agent(env, agent):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    total_reward = 0
    net_worths = []

    for t in range(env.total_steps - env.n_steps):
        with torch.no_grad():
            q_values = agent(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        total_reward += reward
        net_worths.append(env.net_worth)

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

    print(f"Final Net Worth: ${env.net_worth:.2f}")
    print(f"Total Reward from Evaluation: {total_reward:.2f}")

# Evaluate the agent
evaluate_agent(env, agent)
