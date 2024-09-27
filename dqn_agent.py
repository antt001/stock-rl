import torch
import torch.nn as nn

class DQNAgent(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=64, n_layers=1):
        super(DQNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, state):
        # state shape: (batch_size, n_steps, input_size)
        h0 = torch.zeros(self.n_layers, state.size(0), self.hidden_size).to(state.device)
        c0 = torch.zeros(self.n_layers, state.size(0), self.hidden_size).to(state.device)

        out, _ = self.lstm(state, (h0, c0))  # out shape: (batch_size, n_steps, hidden_size)
        out = out[:, -1, :]  # Get the output of the last time step

        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
