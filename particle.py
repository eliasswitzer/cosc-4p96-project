import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MLP

class Particle:
  def __init__(self, search_bounds,init = None):
    self.search_bounds = search_bounds
    num_dims = len(search_bounds)

    if init is None:
      self.position = np.array([np.random.uniform(low, high) for low, high in search_bounds]) # initialize position randomly within bounds
    else:
      self.position = np.array(list(init.values()))
    self.velocity = np.zeros(len(search_bounds)) # initialize velocities to zero

    self.best_position = self.position.copy()
    self.best_fitness = float('inf')

  def get_network_params(self):
    return {
        'num_hidden_layers': int(np.round(self.position[0])),
        'hidden_layer_size': int(np.round(self.position[1])),
        'learning_rate': self.position[2],
        'momentum': self.position[3],
        'batch_size': int(np.round(self.position[4])),
        'weight_decay': self.position[5],
        'dropout_rate': self.position[6]
    }
  
  # evalutate fitness of a particle by training a neural net on the parameters
def evaluate_particle(parameters, train_dataset, val_dataset, test_dataset, input_dim, num_classes, g, epochs=5, overfitting_detection = 10):
  train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True, generator=g)
  val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], generator=g)
  test_loader = DataLoader(test_dataset, batch_size=parameters['batch_size'], generator=g)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MLP(input_dim=input_dim, num_classes=num_classes, num_layers=parameters['num_hidden_layers'], hidden_size=parameters['hidden_layer_size'], dropout_rate=parameters['dropout_rate']).to(device)

  optimizer = torch.optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=parameters['momentum'], weight_decay=parameters['weight_decay'])
  criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([10])) # BCEWithLogitsLoss is used for multi-class classification problems

  running_mean_loss = 0 #stores average loss over minibatches, so this is mean per epoch
  count = 1 #counts minibatches
  loss_history = np.zeros(overfitting_detection)
  loss_history[loss_history == 0.0] = np.nan #do this to do mean_nan - ignores nan values, if we use zeros instead it would messup the mean

  # Training
  model.train()
  for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for images, labels in train_loader:
      images, labels = images.to(device),  labels.float().to(device)
      optimizer.zero_grad()
      loss = criterion(model(images), labels)
      running_mean_loss = running_mean_loss + (loss.item() - running_mean_loss) / count #calculate mean of loss in the batch
      count += 1
      loss.backward()
      optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.float().to(device)
        val_loss += criterion(model(images), labels).item()

    val_loss = val_loss / len(val_loader)
    loss_history[epoch%overfitting_detection] = val_loss

    print("val",val_loss) #debug

    #check overfitting
    mean = np.nanmean(loss_history)
    std = np.nanstd(loss_history)
    if val_loss > mean + std:
        print(f"Overfitting detected at epoch {epoch}, stopping search early.")
        break

  # Testing
  model.eval()
  test_loss = 0.0
  total_eval_metric = 0.0
  with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().to(device)
        test_loss += criterion(model(images), labels).item()
        curr_eval_metric = f1_score(images, labels, model)
        total_eval_metric+= curr_eval_metric

        # print(model(images)[0])
        # print(labels[0])

  print(total_eval_metric)
  avg_test_loss = test_loss / len(test_loader)
  avg_eval_metric = total_eval_metric / len(test_loader)

  print(f"Test Loss: {avg_test_loss:.4f} | Test Evaluation Metric (f1/acc): {avg_eval_metric:.4f}") #debug

  return test_loss / len(test_loader)

"""this function gets f1_score with a given dataset and labels"""
def f1_score(x_set,y_set,model):
  with torch.no_grad(): #this disables gradient calculations

    #sigmoid on each label to get the predictions the model has for each label?
    #crit = nn.Sigmoid()
    outputs = model(x_set)

    #conv to numpy
    numpy_pred = outputs.cpu().numpy()
    y_valid_numpy = y_set.cpu().numpy()

    numpy_pred = np.where(numpy_pred > 0 ,1,0)

    #get tp
    right_guesses = np.where((numpy_pred == y_valid_numpy),1,0) #get all instances where predictions = actual labels
    tp = np.logical_and(right_guesses,y_valid_numpy) #get the predictions where they guess positive (1) label correct
    tp = np.sum(tp.astype(int),axis=1)

    #get fn
    tn = np.logical_not(np.logical_or(numpy_pred,y_valid_numpy)) #get pred where neg (0) label was guessed correct. this is where y_pred=0 and labels = 0, so use nor gate
    tn = np.sum(tn.astype(int),axis=1)

    #false positive - where model guesses positive but its actually negative
    wrong_guesses = np.logical_xor(numpy_pred,y_valid_numpy).astype(int) #instances where predictions != actual labels (1 =wrong)
    fp = np.logical_and(wrong_guesses,numpy_pred) #if model guesses wrong (wrong=1) and pred is 1 (positive), means the label is actually 0 (neg). so when both these are 1 we have a fp
    fp = np.sum(fp.astype(int),axis=1)

    #false negative - model guesses negative but its actually positive
    fn = np.logical_and(wrong_guesses,np.logical_not(numpy_pred)) #if model is wrong, and pred is negative, label is 1. negate the pred label then and it with wrong_guess, this will be one if fn
    fn=  np.sum(fn.astype(int),axis=1)

    #do the calculations if denoms aren't 0, or else just set to 0
    epsilon = 1e-9
    recall = np.where((tp+fn+epsilon) ==0, 0, tp/(tp+fn))
    precision = np.where((tp+fp+epsilon)==0, 0, tp/(tp+fp))
    f1 = np.where((precision+recall) ==0, 0, 2*(precision*recall)/(precision+recall) )

    return  f1.mean()