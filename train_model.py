import torch
from model_lstm import LSTM
n_iterations = 
batch_size = 
printing_gap = 
train_losses = []
test_losses = []
train_loss = []
for iter in range(n_iterations):
    running_loss = 0
    #Get a batch
    train_x, train_y = next_stock_batch(batch_size, n_steps, df_train, )
    #make into tensor
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)

    #make them into torch variables in float format
    train_x = train_x.float()
    train_y = train_y.float()

    #Reset the gradients
    optim.zero_grad()
    
    #Get the outputs
    outputs = model.forward(train_x)

    #detach the hidden state
    
    #compute the loss
    train_loss = criterion(outputs, train_y.flatten())

    #compute the gradients
    train_loss.backward()

    #Apply the gradients
    optim.step()
    running_loss += train_loss.item()
    avg_loss = running_loss/batch_size


    #Append the loss value
    train_losses.append(avg_loss)

    if iter % printing_gap == 0:
        print(f'batch: {iter}')
        #Print the information


plt.plot(train_loss, label= "Train Loss")
plt.xlabel(" Iteration ")
plt.ylabel("Loss value")
plt.legend(loc="upper left")
plt.show()