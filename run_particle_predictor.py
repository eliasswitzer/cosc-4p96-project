import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import ParticlePredictor
from data import training_data

"""This class contains the particle predictor"""
"""it uses stored particle representations with validation accuracy to train a simple ANN with"""

#some parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500

#define some nn stuff
model = ParticlePredictor().to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#grab and format imported training data
train_set = np.array(list(training_data[0][0].values())) #do one pass manually to initialize variables
train_labels = np.array(training_data[0][1])
for i in range(1,len(training_data)): #append everything in a 1d array
    train_set = np.append(train_set, np.array(list(training_data[i][0].values())))
    train_labels = np.append(train_labels, np.array(training_data[i][1]))

#more manipulations
train_set = train_set.reshape(len(training_data),7) #reformat
train_set  = (train_set - train_set.mean(axis=0))/train_set.std(axis=0)#zscore normalize

#create tensors
train_labels_tensor = torch.tensor(train_labels.reshape(-1,1), dtype=torch.float32)
train_set_tensor = torch.tensor(train_set, dtype=torch.float32)

#load pytorch stuff
dataset = TensorDataset(train_set_tensor,train_labels_tensor) #create dataset out of our tensors
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

#FORWARD PASS
for epoch in range(EPOCHS):
    model.train()
    avg_loss = 0.0

    #loads minibatch
    for x,y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        #forward pass and loss
        loss = criterion(model(x), y)
        avg_loss +=loss.item()

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch==0 or epoch%100==0):
        print(f"Average Loss: {avg_loss/len(dataloader)}")

# debug: print out some of the labels
for x, y in DataLoader(dataset, batch_size=20):
    x, y = x.to(DEVICE), y.to(DEVICE)
    outputs = model(x)
    print(outputs)
    break
