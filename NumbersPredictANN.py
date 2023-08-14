import numpy as np
import torch
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F

# read in data
dataset = np.genfromtxt("MNIST_DATA.csv", delimiter=",")

dataset = np.delete(dataset, 0, 0) # delete column names from numpy array

y = dataset[:, 0:1] # set dependent variable
X = dataset[:, 1:] # set independent variables

torch.manual_seed(0)

# convert to pytorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64).reshape(-1,1)
y = y.view(-1)

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
        x = torch.flatten(x, 1) # flaatten image
        x = self.model(x) # apply network to our input (x) batch
        return x # return predictions

model = MyNetwork()
print(model)

# set batch and epocjs
BATCH_SIZE = 50
EPOCHS = 10

# create loss function
loss_function = nn.NLLLoss()
# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=.003)

# train model 

for epochs in range(EPOCHS):
    for batch in range(0, len(X), BATCH_SIZE):
        # grab batch of samples
        X_samples = X[batch:batch+BATCH_SIZE]
        Y_samples = y[batch:batch+BATCH_SIZE]
        
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
    y_predictions = model(X)
    predictions_adjusted = torch.exp(y_predictions) # adjust numbers back after LogSoftmax
    predictions = torch.argmax(predictions_adjusted, dim=1)

accuracy = ((predictions == y)).sum() / len(y)
print(f"Accuracy: {accuracy:>7f}")