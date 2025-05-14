import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x):
        # Attention mechanism
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed forward network
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class DQNAgent(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128,num_channels=64, n_layers=2):
        super(DQNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout_conv = nn.Dropout(p=0.2)
        # Transformer layer
        self.transformer = TransformerBlock(num_channels, hidden_size, n_layers)

        # Fully connected layers
        self.fc1 = nn.Linear(num_channels, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        # Transpose for Conv1d: (batch_size, input_size, n_steps)
        x = state.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x = self.dropout_conv(self.relu(self.conv2(x)))
        x = self.dropout_conv(self.relu(self.conv3(x)))
        x = self.dropout_conv(self.relu(self.conv4(x)))
        # Transpose back: (batch_size, n_steps, num_channels)
        x = x.permute(0, 2, 1)

        # Pass through transformer
        x = self.transformer(x)

        out = self.relu(self.fc1(x))
        out = self.dropout_conv(out)
        out = self.fc2(out)
        return out
