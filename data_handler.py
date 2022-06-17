import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt 


data_X_path  = "C:/Users/Omistaja/Desktop/epicode/rnn-activity-recognition/Train/X_train.txt"
data_y_path  = "C:/Users/Omistaja/Desktop/epicode/rnn-activity-recognition/Train/y_train.txt"
feature_info = "C:/Users/Omistaja/Desktop/epicode/rnn-activity-recognition/features.txt"

data_X = pd.read_csv(data_X_path, delim_whitespace=' ', header=None)
data_y = pd.read_csv(data_y_path,  header=None)
feature_info_names = pd.read_csv(feature_info, header=None)

data_y.columns = ['Activity']

feature_list = []
for i in feature_info_names[0]:
     feature_list.append(i.strip())
# print(data.head(5))
# data_X
# feature_info_names

selected_feature = []

# feature_columns = feature_list 
data_X.columns = feature_list
# print(feature_list)
for i in  data_X.columns:
     if i.endswith('Mean-1'):
          # print(i)
          selected_feature.append(i)

# print(selected_feature)

selected_X = data_X[selected_feature]


selected_y = data_y 
# print(selected_X)
# print(selected_y)

df = pd.concat([selected_X, selected_y], axis=1)
df = df.dropna()
# df.to_csv('processed_data.csv')





def next_stock_batch(batch_size, n_steps, df_base):

    features = df_base.iloc[:, :].values 
    starting_point = np.random.randint(0, len(df_base)-n_steps-1,  batch_size)

    x, y = [], []

    for i in starting_point:
        sequence = features[i:i+n_steps-1,  :]
        target   = features[i+1:i+n_steps+1, -1]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)



# batch_size, n_steps, df_base  = 64, 50, df


# x, y = next_stock_batch(batch_size, n_steps, df_base)

# print(x.shape)
# print(y.shape)




class Recurrent_Neural_Network(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers, batch_size, seq_length):
        super(Recurrent_Neural_Network, self).__init__()
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



in_size, hid_size, n_layers, batch_size, seq_length = 18, 10, 15, 50, 32 

model = Recurrent_Neural_Network(in_size, hid_size, n_layers, batch_size, seq_length)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003) 


all_train_losses, all_test_losses, epochs = [], [], 1000

for epoch in range(epochs):
    train_losses = 0 
    features_train, target_train = next_stock_batch(batch_size, seq_length, df)
    features_train, target_train = torch.tensor(features_train).float(), torch.tensor(target_train).float()

    optimizer.zero_grad()
    prediction_train = model.forward(features_train)
    loss_train = criterion(prediction_train, target_train)
    loss_train.backward()
    optimizer.step()

    train_losses += loss_train.item()
    all_train_losses.append(train_losses/len(features_train))

    model.eval()
    with torch.no_grad():
        test_losses = 0 
        features_test, target_test = next_stock_batch(batch_size, seq_length, df)
        features_test, target_test = torch.tensor(features_test).float(), torch.tensor(target_test).float()

        prediction_test = model.forward(features_test)
        loss_test = criterion(prediction_test, target_test)
        test_losses += loss_test.item()
        all_test_losses.append(test_losses/len(features_test))

    model.train()

    print(f'Train loss :    {train_losses:.6f}     |     Test Loss  : {test_losses:.6f}')


plt.plot(all_train_losses, label='train loss') 
plt.plot(all_test_losses,  label='test loss') 
plt.legend()
plt.show()




































'tBodyAcc-Mean-1'
'tGravityAcc-Mean-1'
'tBodyAccJerk-Mean-1'
'tBodyGyro-Mean-1'
'tBodyGyroJerk-Mean-1'
'tBodyAccMag-Mean-1'
'tGravityAccMag-Mean-1'
'tBodyAccJerkMag-Mean-1'
'tBodyGyroMag-Mean-1'
'tBodyGyroJerkMag-Mean-1'
'fBodyAcc-XYZ-Mean-1'
'fBodyAccJerk-XYZ-Mean-1'
'fBodyGyro-XYZ-Mean-1'
'fBodyAccMag-Mean-1'
'fBodyAccJerkMag-Mean-1'
'fBodyGyroMag-Mean-1'
'fBodyGyroJerkMag-Mean-1'
