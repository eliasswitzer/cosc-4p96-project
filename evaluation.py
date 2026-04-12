import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models import MLP

# evalutate fitness of a particle by training a neural net on the parameters
def evaluate_particle(parameters, train_dataset, val_dataset, input_dim, num_classes, g, epochs=5, overfitting_detection = 10):
  train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True, generator=g)
  val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], generator=g)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MLP(input_dim=input_dim, num_classes=num_classes, num_layers=parameters['num_hidden_layers'], hidden_size=parameters['hidden_layer_size'], dropout_rate=parameters['dropout_rate']).to(device)

  optimizer = torch.optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=parameters['momentum'], weight_decay=parameters['weight_decay'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  single_label = "multi-label" not in (train_dataset.info['task']) #true if single label

  # Determine the right loss function for the job
  if single_label:
    criterion = nn.CrossEntropyLoss()
  else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device)) # BCEWithLogitsLoss is used for multi-class classification problems

  loss_history = np.full(overfitting_detection, np.nan)
  val_f1 = 0.0
  evaluation_metrics = [0,0] #storing two of the evaluation metrics
  for epoch in range(epochs):
    # Training
    model.train()
    for images, labels in train_loader:

      if single_label:
        images, labels = images.to(device), labels.squeeze(1).long().to(device)
      else:
        images, labels = images.to(device),  labels.float().to(device) #for multi label
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
    total_acc = 0.0
    avg_acc = 0.0

    with torch.no_grad():
      for images, labels in val_loader:
        if single_label:
          images, labels = images.to(device), labels.squeeze(1).long().to(device)
        else:
          images, labels = images.to(device),  labels.float().to(device) #for multi label
        outputs = model(images)
        val_loss += criterion(model(images), labels).item()

        #for multi-class
        if single_label:
          y_pred = outputs.argmax(1)
          acc = y_pred == labels
          total_acc += acc.cpu().numpy().astype(int).sum()/len(acc)

        else:
          tp, fp, fn = get_batch_metrics(outputs, labels)
          total_tp += tp
          total_fp += fp
          total_fn += fn
          ham_loss += np.mean(hamming_loss(outputs,labels))

    avg_val_loss = val_loss / len(val_loader)

    if single_label:
      avg_acc = total_acc / len(val_loader)

      #when doing multi class we want to return thse
      evaluation_metrics[0]+=avg_acc
      evaluation_metrics[1]+=avg_acc #TODO: get f1 score working for this

    else:
      loss_history[epoch%overfitting_detection] = avg_val_loss
      val_f1 = f1_score(total_tp, total_fp, total_fn)
      ham_loss = ham_loss / len(val_loader)

      #when doing multi label we want to return these
      evaluation_metrics[0]+=val_f1
      evaluation_metrics[1]+=ham_loss

    print(f"Epoch {epoch+1}/{epochs} | Validation Loss: {avg_val_loss} | Validation F1: {val_f1} | Hamming Loss: {ham_loss} | Acc: {avg_acc}")

    #check overfitting
    if not np.all(np.isnan(loss_history)):
        mean = np.nanmean(loss_history)
        std = np.nanstd(loss_history)
        if avg_val_loss > mean + std and epoch > 0:
            print(f"Overfitting detected at epoch {epoch}, stopping search early.")
            break

  return evaluation_metrics[0]/epochs, evaluation_metrics[1]/epochs

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
  single_label = "multi-label" not in (test_dataset.info['task']) #true if single label

  # Determine the right loss function for the job
  if single_label:
    criterion = nn.CrossEntropyLoss()
  else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device)) # BCEWithLogitsLoss is used for multi-class classification problems


  train_loader = DataLoader(train_dataset, batch_size=best_parameters['batch_size'], shuffle=True, generator=g)
  test_loader = DataLoader(test_dataset, batch_size=best_parameters['batch_size'])

  print("Training Final Model on Best Found Parameters")

  for epoch in range(epochs):
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}")
    model.train()
    for images, labels in train_loader:
      if single_label:
        images, labels = images.to(device), labels.squeeze(1).long().to(device)
      else:
        images, labels = images.to(device),  labels.float().to(device) #for multi label
      optimizer.zero_grad()
      loss = criterion(model(images), labels)
      loss.backward()
      optimizer.step()
    scheduler.step()

  # Testing
  model.eval()
  total_tp, total_fp, total_fn = 0, 0, 0
  ham_loss = 0.0
  total_acc = 0.0
  test_f1 =0.0
  avg_acc = 0.0
  test_auc = 0.0

  all_probs = []
  all_targets = []

  with torch.no_grad():
    for images, labels in test_loader:
      #do the right preprocessing to the labels/images
      if single_label:
        images, labels = images.to(device), labels.squeeze(1).long().to(device)
      else:
        images, labels = images.to(device),  labels.float().to(device) #for multi label
      outputs = model(images)

      # store predictions for AUC
      if single_label:
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
      else:
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

      #for multi-class
      if single_label:
        y_pred = outputs.argmax(1)
        acc = y_pred == labels
        total_acc += acc.cpu().numpy().astype(int).sum()/len(acc)
      else:
        tp, fp, fn = get_batch_metrics(outputs, labels)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ham_loss += np.mean(hamming_loss(outputs,labels))

  evaluation_metrics = [] #append metrics to a list of things to return

  # Format for AUC calculation
  all_probs = np.vstack(all_probs)
  all_targets = np.concatenate(all_targets) if single_label else np.vstack(all_targets)

  if single_label:
    avg_acc = total_acc / len(test_loader)

    try:
      test_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except ValueError:
      test_auc = 0.0

    #when doing multi class we want to return thse
    evaluation_metrics.append(avg_acc)
    evaluation_metrics.append(avg_acc) #TODO: get f1 score working for this

  else:
    test_f1 = f1_score(total_tp, total_fp, total_fn)
    ham_loss = ham_loss / len(test_loader)

    try:
      test_auc = roc_auc_score(all_targets, all_probs, average='macro')
    except ValueError:
      test_auc = 0.0

    #when doing multi label we want to return these
    evaluation_metrics.append(test_f1)
    evaluation_metrics.append(ham_loss)
  print(f"Epoch {epoch+1}/{epochs} | Validation F1: {test_f1} | Hamming Loss: {ham_loss} | Acc: {avg_acc}")

  print(f"Final Test Evaluation Metrics: F1: {test_f1:.4f}  | Hamming Loss: {ham_loss:.4f} | Acc: {avg_acc:.4f} | AUC: {test_auc:.4f} ")
  return evaluation_metrics[0], evaluation_metrics[1]

# a simplier version of the evaluate particle function for convienently collecting data about particles for training the predictor
def collect_particle_data(best_parameters, train_dataset, test_dataset, input_dim, num_classes, g):
  """Trains the best found architecture for more epochs and evaluates on the test set"""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Initialize the best model
  model = MLP(input_dim=input_dim, num_classes=num_classes, num_layers=best_parameters['num_hidden_layers'], hidden_size=best_parameters['hidden_layer_size'], dropout_rate=best_parameters['dropout_rate']).to(device)

  epochs = 10
  optimizer = torch.optim.SGD(model.parameters(), lr=best_parameters['learning_rate'], momentum=best_parameters['momentum'], weight_decay=best_parameters['weight_decay'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  criterion = nn.CrossEntropyLoss()

  train_loader = DataLoader(train_dataset, batch_size=best_parameters['batch_size'], shuffle=True, generator=g)
  test_loader = DataLoader(test_dataset, batch_size=best_parameters['batch_size'])

  for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
      images, labels = images.to(device), labels.squeeze(1).to(device)

      optimizer.zero_grad()
      loss = criterion(model(images), labels)
      loss.backward()
      optimizer.step()
    scheduler.step()

  # Testing
  model.eval()
  total_acc = 0.0
  avg_acc = 0.0

  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.squeeze(1).to(device)

      outputs = model(images)
      y_pred = outputs.argmax(1)

      acc = y_pred == labels
      total_acc += acc.cpu().numpy().astype(int).sum()/len(acc)

  avg_acc = total_acc / len(test_loader)

  #prints in a format easy to copy paste
  print(f"[{best_parameters},{avg_acc:.4f}],")
