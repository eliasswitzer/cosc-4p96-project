import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MLP

# evalutate fitness of a particle by training a neural net on the parameters
def evaluate_particle(parameters, train_dataset, val_dataset, input_dim, num_classes, g, epochs=5, overfitting_detection = 10):
  train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True, generator=g)
  val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], generator=g)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MLP(input_dim=input_dim, num_classes=num_classes, num_layers=parameters['num_hidden_layers'], hidden_size=parameters['hidden_layer_size'], dropout_rate=parameters['dropout_rate']).to(device)

  optimizer = torch.optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=parameters['momentum'], weight_decay=parameters['weight_decay'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device)) # BCEWithLogitsLoss is used for multi-class classification problems

  loss_history = np.full(overfitting_detection, np.nan)
  val_f1 = 0.0

  for epoch in range(epochs):
    # Training
    model.train()
    for images, labels in train_loader:
      images, labels = images.to(device),  labels.float().to(device)
      optimizer.zero_grad()
      loss = criterion(model(images), labels)
      loss.backward()
      optimizer.step()

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    ham_loss = 0.0

    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images)
        val_loss += criterion(model(images), labels).item()

        tp, fp, fn = get_batch_metrics(outputs, labels)
        total_tp += tp
        total_fp += fp
        total_fn += fn


        ham_loss += np.mean(hamming_loss(outputs,labels))

    avg_val_loss = val_loss / len(val_loader)
    loss_history[epoch%overfitting_detection] = avg_val_loss
    val_f1 = f1_score(total_tp, total_fp, total_fn)
    ham_loss = ham_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Validation Loss: {avg_val_loss} | Validation F1: {val_f1} | Hamming Loss: {ham_loss}")

    #check overfitting
    if not np.all(np.isnan(loss_history)):
        mean = np.nanmean(loss_history)
        std = np.nanstd(loss_history)
        if avg_val_loss > mean + std and epoch > 0:
            print(f"Overfitting detected at epoch {epoch}, stopping search early.")
            break

  return val_f1, ham_loss

def get_batch_metrics(outputs, labels):
  """Helper function to get raw TP, FP, and FN counts from a batch"""
  preds = (outputs > 0).float()
  tp = torch.sum((preds == 1) & (labels == 1)).item()
  fp = torch.sum((preds == 1) & (labels == 0)).item()
  fn = torch.sum((preds == 0) & (labels == 1)).item()
  return tp, fp, fn

def hamming_loss(outputs,labels):
  """helper function that calculates hamming distance of labels from the current batch"""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  preds = (outputs > 0).float()
  #numerator here is hamming distance, divide it by number of labels to get hamming loss
  return (np.sum(np.logical_xor(preds.cpu().numpy(), labels.cpu().numpy()),axis=1))/len(labels[0])


def f1_score(tp, fp, fn):
  """Computes F1-score given number of true positives, false positives, and false negatives"""
  epsilon = 1e-9 # added to prevent division by 0
  precision = tp / (tp + fp + epsilon)
  recall = tp / (tp + fn + epsilon)
  f1 = 2 * (precision * recall) / (precision + recall + epsilon)
  return f1

def test_model(best_parameters, train_dataset, test_dataset, input_dim, num_classes, g):
  """Trains the best found architecture for more epochs and evaluates on the test set"""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Initialize the best model
  model = MLP(input_dim=input_dim, num_classes=num_classes, num_layers=best_parameters['num_hidden_layers'], hidden_size=best_parameters['hidden_layer_size'], dropout_rate=best_parameters['dropout_rate']).to(device)

  epochs = 50
  optimizer = torch.optim.SGD(model.parameters(), lr=best_parameters['learning_rate'], momentum=best_parameters['momentum'], weight_decay=best_parameters['weight_decay'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([10.0])).to(device)

  train_loader = DataLoader(train_dataset, batch_size=best_parameters['batch_size'], shuffle=True, generator=g)
  test_loader = DataLoader(test_dataset, batch_size=best_parameters['batch_size'])

  print("Training Final Model on Best Found Parameters")

  for epoch in range(epochs):
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}")
    model.train()
    for images, labels in train_loader:
      images, labels = images.to(device),  labels.float().to(device)
      optimizer.zero_grad()
      loss = criterion(model(images), labels)
      loss.backward()
      optimizer.step()
    scheduler.step()
  
  model.eval()
  total_tp, total_fp, total_fn = 0, 0, 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.float().to(device)
      outputs = model(images)
      tp, fp, fn = get_batch_metrics(outputs, labels)
      total_tp += tp
      total_fp += fp
      total_fn += fn

  test_f1 = f1_score(total_tp, total_fp, total_fn)
  print(f"Final Test F1-Score: {test_f1:.4f}")
  return test_f1

