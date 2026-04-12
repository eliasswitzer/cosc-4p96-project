import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import ParticlePredictor
from data import training_data,testing_data
from random import randint

"""This class contains the particle predictor"""
"""it uses stored particle representations with validation accuracy to train a simple ANN with"""

#some parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300

#custom loss function
def my_loss(output, target):
    SE = (output - target)**2
    SE = torch.where((target<=.8) ,SE*1.2,SE)
    SE= torch.where((target<=.7) ,SE*1.4,SE) #these should compound
    SE= torch.where((target<=.6) ,SE*1.6,SE)
    SE= torch.where((target<=.5) ,SE*1.8,SE)

    return torch.mean(SE)

#define some nn stuff
model = ParticlePredictor().to(DEVICE)
criterion =   torch.nn.MSELoss() #my_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#grab and format imported training data
num_data = 1
train_set = np.array(list(training_data[0][0].values())) #do one pass manually to initialize variables
train_labels = np.array(training_data[0][1])
for i in range(1,len(training_data)): #append everything in a 1d array
    #if training_data[i][1] <= 0.8 or randint(0,1) ==0: #50% chance to accept data over 80%
        # if training_data[i][1] <= 0.5:  #set labels under a threshold to 0.5 to prevent model from trying hard to understand worthless data
        #     training_data[i][1] = 0.5
        num_data+=1
        train_set = np.append(train_set, np.array(list(training_data[i][0].values())))
        train_labels = np.append(train_labels, np.array(training_data[i][1]))

#more manipulations
train_set = train_set.reshape(num_data,7) #reformat
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

model = ParticlePredictor()
model.load_state_dict(torch.load("model", weights_only=True))
model.eval()

#grab and format imported testing data
test_set = np.array(list(testing_data[0][0].values())) #do one pass manually to initialize variables
test_labels = np.array(testing_data[0][1])
for i in range(1,len(testing_data)): #append everything in a 1d array
    test_set = np.append(test_set, np.array(list(testing_data[i][0].values())))
    test_labels = np.append(test_labels, np.array(testing_data[i][1]))

#more manipulations
test_set = test_set.reshape(len(testing_data),7) #reformat
test_set  = (test_set - test_set.mean(axis=0))/test_set.std(axis=0)#zscore normalize

#create tensors
test_labels_tensor = torch.tensor(test_labels.reshape(-1,1), dtype=torch.float32)
test_set_tensor = torch.tensor(test_set, dtype=torch.float32)

#save model
#torch.save(model.state_dict(),"model")

# debug: print out some of the labels
model.eval()
avg_loss = 0
last_outputs = 0
for x, y in DataLoader(TensorDataset(test_set_tensor,test_labels_tensor), batch_size=20):
    x, y = x.to(DEVICE), y.to(DEVICE)
    outputs = model(x)
    loss = criterion(outputs, y)
    avg_loss +=loss.item()
    last_outputs = outputs
print(avg_loss/25)
print(last_outputs)
