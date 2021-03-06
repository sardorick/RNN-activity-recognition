import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# define the Recurrent Neural Network

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, seq_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # linear layers
        self.fc1 = nn.Linear(self.batch_size*self.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, self.batch_size)

    def forward(self, x):
        h_0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_size))
        rnn_out, h_n = self.rnn(x, h_0)
        last_hidden = h_n[-1]
        x = F.relu(last_hidden.flatten())
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

# test the model 
