def evaluate_agent(env, agent):
    state = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    net_worths = []

    for t in range(env.total_steps):
        with torch.no_grad():
            q_values = agent(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        state = torch.FloatTensor(next_state)
        total_reward += reward
        net_worths.append(env.net_worth)

        if done:
            break

    # Plotting net worth over time
    plt.plot(net_worths)
    plt.title('Agent Net Worth Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Worth')
    plt.show()
