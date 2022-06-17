import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, seq_len):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_len = self.seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)

        self.fc1 = nn.Linear((self.num_layers, self.batch_size, self.hidden_size))
        self.fc2 = nn.Linear((self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        h_0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_size))
        c_0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_size))
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        last_hidden = h_n[-1]
        print(last_hidden.shape)

        x = F.relu(last_hidden.flatten())
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
    