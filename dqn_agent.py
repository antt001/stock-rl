import torch
import torch.nn as nn

class DQNAgent(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128,num_channels=64, n_layers=2):
        super(DQNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # LSTM layer
        self.lstm = nn.LSTM(num_channels, hidden_size, n_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        # Transpose for Conv1d: (batch_size, input_size, n_steps)
        x = state.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Transpose back: (batch_size, n_steps, num_channels)
        x = x.permute(0, 2, 1)

        # state shape: (batch_size, n_steps, input_size)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(state.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(state.device)

        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, n_steps, hidden_size)
        out = out[:, -1, :]  # Get the output of the last time step

        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
