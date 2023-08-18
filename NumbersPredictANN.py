import numpy as np
import torch
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# read in data
dataset = np.genfromtxt("MNIST_DATA.csv", delimiter=",")

dataset = np.delete(dataset, 0, 0) # delete column names from numpy array

y = dataset[:, 0:1] # set dependent variable
X = dataset[:, 1:] # set independent variables

torch.manual_seed(0)

# convert to pytorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64).reshape(-1,1)

# define train and test sizes
train_size = int(0.8*len(X))
test_size = len(X) - train_size
y = y.view(-1)

X_train, X_test = torch.utils.data.random_split(X, [train_size, test_size])
y_train, y_test = torch.utils.data.random_split(y, [train_size, test_size])

class MyNetwork(nn.Module):
    def __init__(self):
        # calls super class of nn.module before we do our own initialization
        super(MyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 392),
            nn.ReLU(True),
            nn.Linear(392, 196),
            nn.ReLU(True),
            nn.Linear(196, 10), 
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten image
        x = self.model(x) # apply network to our input (x) batch
        return x # return predictions
        
model = MyNetwork()
print(model)

# set batch and epochs
BATCH_SIZE = 50
EPOCHS = 15

# create loss function
loss_function = nn.NLLLoss()
# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=.005)

# train model 

for epochs in range(EPOCHS):
    for batch in range(0, len(X_train), BATCH_SIZE):
        # grab batch of samples
        X_samples = X_train[batch:batch+BATCH_SIZE]
        Y_samples = y_train[batch:batch+BATCH_SIZE]
        
        # create predictions
        predictions = model(X_samples)
        
        # calculate loss
        loss = loss_function(predictions, Y_samples)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss: {loss:>7f} [{epochs}]")

# begin computation of accuracy
with torch.no_grad():
    X_samples_ = X_test[0: 0+test_size-1]
    y_predictions = model(X_samples_)
    predictions_adjusted = torch.exp(y_predictions) # adjust numbers back after LogSoftmax
    predictions = torch.argmax(predictions_adjusted, dim=1)

Y_samples_ = y_test[0: 0+test_size-1]
accuracy = ((predictions == Y_samples_)).sum() / len(y_test) 
print(f"Accuracy: {accuracy:>7f}")