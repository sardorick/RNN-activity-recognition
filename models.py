
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Recurrent_Neural_Network_1(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers, batch_size, seq_length):
        super(Recurrent_Neural_Network_1, self).__init__()
        self.batch_size = batch_size 
        self.n_layers   = n_layers 
        self.hidden_size = hidden_size 
        self.rnn = nn.RNN(in_size, hidden_size, n_layers, batch_first=True)


        self.fully_connected  = nn.Linear(hidden_size, seq_length)

    def forward(self, x):
        h_0 = torch.zeros((self.n_layers, self.batch_size, self.hidden_size))
        _, h_n = self.rnn(x, h_0)
        last_hidden = h_n[-1]
        output = F.relu(self.fully_connected(last_hidden))
        return output 















class Recurrent_Neural_Network_2(nn.Module):
     def __init__(self, in_size, hidden_size, batch_size, num_layers, seq_len):
          super().__init__()
          self.input_size = in_size
          self.hidden_size = hidden_size
          self.num_layers = num_layers
          self.batch_size = batch_size
          self.seq_len = seq_len
          self.rnn = nn.RNN(in_size, hidden_size, num_layers, batch_first=True)

          # linear layers
          # self.fc1 = nn.Linear(self.batch_size*self.hidden_size, 1024)
          self.fc1 = nn.Linear(self.hidden_size, 1024)
          self.fc2 = nn.Linear(1024, seq_len)

     def forward(self, x):
          h_0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_size))
          rnn_out, h_n = self.rnn(x, h_0)
          last_hidden = h_n[-1]
          x = F.relu(last_hidden.flatten())
          x = F.relu(self.fc1(x))
          out = self.fc2(x)
          return out