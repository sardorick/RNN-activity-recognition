import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from models import Recurrent_Neural_Network_1, Recurrent_Neural_Network_2


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







in_size, hid_size, n_layers, batch_size, seq_length = 18, 20, 25, 64, 100
model = Recurrent_Neural_Network_1(in_size, hid_size, n_layers, batch_size, seq_length)

# in_size, hidden_size, batch_size, num_layers, seq_length = 18, 10, 32, 15, 100
# model = Recurrent_Neural_Network_2(in_size, hidden_size, batch_size, num_layers, seq_length)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003) 


all_train_losses, all_test_losses, epochs = [], [], 250

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
    average_train_losses = train_losses/len(features_train)
    all_train_losses.append(average_train_losses)

    model.eval()
    with torch.no_grad():
        test_losses = 0 
        features_test, target_test = next_stock_batch(batch_size, seq_length, df)
        features_test, target_test = torch.tensor(features_test).float(), torch.tensor(target_test).float()

        prediction_test = model.forward(features_test)
        loss_test = criterion(prediction_test, target_test)
        test_losses += loss_test.item()
        average_test_losses = test_losses/len(features_test)
        all_test_losses.append(average_test_losses)

    model.train()

    print(f' {epoch+1:3}/{epochs}  Train loss :    {average_train_losses:.10f}     |     Test Loss  : {average_test_losses:.10f}')


torch.save({ "model_state": model.state_dict(), 'in_size' : 18, 'hid_size' : 20, 'n_layers' : 25, 'batch_size' : 64, 'seq_length' : 500 }, 'trained_model')

plt.plot(all_train_losses, label='train loss') 
plt.plot(all_test_losses,  label='test loss') 
plt.legend()
plt.show()
















