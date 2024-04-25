import torch
import torch.nn as nn
import numpy as np

import json
import math
from tqdm import tqdm
from torchinfo import summary
from data_loader.data_loaders import *
from model.loss import weighted_CrossEntropyLoss
from model.metric import *
from model.model import CCT
from model.attn_model import AttnSleep
from utils.util import load_folds_data, calc_class_weight


seed = 2024
torch.manual_seed(seed=seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(seed=seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np_data_dir = r'dataset\edf-20'


def adjust_learning_rate(optimizer, epoch, warmup=False, warmup_ep=0, enable_cos=True):
    lr = 3e-4
    if warmup and epoch < warmup_ep:
        lr = lr / (warmup_ep - epoch)
    elif enable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_ep) / (150 - warmup_ep)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(model, epoch, total_epochs, data_loader, optimizer, criterion,
                cls_weight, valid_data_loader):
    model.train()
    
    # Initialize accumulators for train accuracy, loss and f1 score
    train_acc, train_loss, train_f1 = 0, 0, 0
    
    # train
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target, cls_weight, device)
        loss.backward()
        
        optimizer.step()
        
        acc = accuracy(output, target)
        f1_score = f1(output, target)
        
        # Accumulate the accuracy, loss and f1 score for each batch
        # print(acc, loss, f1_score)
        train_acc += acc
        train_loss += loss.item()
        train_f1 += f1_score
        
    # Calculate average accuracy, loss and f1 score over all training batches
    train_acc /= len(data_loader)
    train_loss /= len(data_loader)
    train_f1 /= len(data_loader)
    
    model.eval()
    
    # Initialize accumulators for validation accuracy, loss and f1 score
    valid_acc, valid_loss, valid_f1 = 0, 0, 0
    
    with torch.no_grad():
        
        for batch_idx, (data, target) in enumerate(valid_data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target, cls_weight, device)
            
            acc = accuracy(output, target)
            f1_score = f1(output, target)
            
            # Accumulate the accuracy, loss and f1 score for each batch
            valid_acc += acc
            valid_loss += loss.item()
            valid_f1 += f1_score
            
    # Calculate average accuracy, loss and f1 score over all validation batches
    valid_acc /= len(valid_data_loader)
    valid_loss /= len(valid_data_loader)
    valid_f1 /= len(valid_data_loader)
    
    print(f'Epoch: {epoch}/{total_epochs}, Train Acc: {train_acc:.2f}, Train Loss: {train_loss:.2f}, Train F1: {train_f1:.2f}, Valid Acc: {valid_acc:.2f}, Valid Loss: {valid_loss:.2f}, Valid F1: {valid_f1:.2f}')
    return train_acc, train_loss, valid_acc, valid_f1, valid_loss


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, test_data_loader, folder_path, config_name, test_sub_num, target_names=None):
  """
  Evaluates the model and plots the confusion matrix.

  Args:
      model: The model to evaluate.
      test_data_loader: The test data loader.
      target_names: Optional list of target class names (used for confusion matrix labels).

  Returns:
      test_acc: Average accuracy over all test batches.
      test_f1: Average F1 score over all test batches.
  """

  model.eval()

  # Initialize accumulators for test accuracy, loss and f1 score
  test_acc, test_f1 = 0, 0
  all_predictions, all_targets = [], []

  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_data_loader):
      data, target = data.to(device), target.to(device)
      output = model(data)

      acc = accuracy(output, target)
      f1_score = f1(output, target)

      # Get predictions and targets for confusion matrix
      predictions = torch.argmax(output, dim=1)  # Assuming output is logits/probabilities
      all_predictions.extend(predictions.cpu().numpy())
      all_targets.extend(target.cpu().numpy())

      # Accumulate the accuracy, loss and f1 score for each batch
      test_acc += acc
      test_f1 += f1_score

  # Calculate average accuracy, loss and f1 score over all test batches
  test_acc /= len(test_data_loader)
  test_f1 /= len(test_data_loader)

  # Plot and save confusion matrix
  cm = confusion_matrix(all_targets, all_predictions)

  plt.figure(figsize=(8, 6))
  plt.imshow(cm, cmap='Blues')
  plt.colorbar()

  if target_names:
    plt.xticks(range(len(target_names)), target_names, rotation=45)
    plt.yticks(range(len(target_names)), target_names)
  else:
    plt.xticks(range(cm.shape[0]))
    plt.yticks(range(cm.shape[1]))

  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix')
  plt.tight_layout()

  # Save the confusion matrix as a PNG image (you can change the format)
  plt.savefig('{}/conf_matrix/{}_{}_'.format(folder_path, config_name, test_sub_num) + 'confusion_matrix.png')
  plt.close()  # Close the plot window

  return test_acc, test_f1


def plot_training_results(train_acc, train_loss, valid_acc, valid_loss, config_name, test_sub_num, epochs, folder_path):
  """Plots training and validation accuracy and loss curves.

  Args:
    epochs: List of epoch numbers.
    train_acc: List of training accuracy values.
    train_loss: List of training loss values.
    valid_acc: List of validation accuracy values.
    valid_loss: List of validation loss values.
  """
  epochs = [i for i in range(1, epochs+1)]
#   print(epochs)

  # Plot accuracy
  plt.figure(figsize=(10, 6))
  plt.plot(epochs, train_acc, label='Training Accuracy')
  plt.plot(epochs, valid_acc, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.legend()
  plt.grid(True)
  plt.savefig('{}/accs_graphs/{}_{}_'.format(folder_path, config_name, test_sub_num) + 'accuracy_plot.png')

  # Plot loss
  plt.figure(figsize=(10, 6))
  plt.plot(epochs, train_loss, label='Training Loss')
  plt.plot(epochs, valid_loss, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.grid(True)
  plt.savefig('{}/loss_graphs/{}_{}_'.format(folder_path, config_name, test_sub_num) + 'loss_plot.png')


import csv

def save_to_csv(data, filename):
  with open(filename, 'w', newline='') as csvfile:
    fieldnames = ['test_sub_num', 'accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for subject_num, accuracy in data.items():
      writer.writerow({'test_sub_num': subject_num, 'accuracy': accuracy})


def create_dir(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory: {error}")


def main():
    # load config parameters
    config_name = 'attn_sleep_config' #'attn_sleep_config'
    try:
        with open(config_name+'.json', 'r') as jsonfile:
            config = json.load(jsonfile)
    except:
        print("no such config exists")
    
    
    # Define the directory path
    folder_path = "checkpoints/{}_results/".format(config_name)
    conf_matrix_path = folder_path + 'conf_matrix/'
    acc_graphs_path = folder_path + 'accs_graphs/'
    loss_graphs_path = folder_path + 'loss_graphs/'


    create_dir(folder_path)
    create_dir(conf_matrix_path)
    create_dir(acc_graphs_path)
    create_dir(loss_graphs_path)
    
    # criterion = weighted_CrossEntropyLoss
    metrics = [accuracy, f1]
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    batch_size = 128
    epochs = 150
    
    folds_data = load_folds_data(np_data_dir, 20)
    
    # print(config['kernel_sizes'])
    
    results = dict()
    
    for fold_id in tqdm(range(20)):        
        # model = CCT(kernel_sizes=config['kernel_sizes'], stride=config['stride'], padding=config['padding'],
        # pooling_kernel_size=config['pooling_kernel_size'], pooling_stride=config['pooling_stride'], 
        # pooling_padding=config['pooling_padding'], n_conv_layers=config['n_conv_layers'], 
        # n_input_channels=config['n_input_channels'], in_planes=config['in_planes'], activation=config['activation'], # ReLU
        # max_pool=config['max_pool'], conv_bias=config['conv_bias'], dim=config['dim'], num_layers=config['num_layers'],
        # num_heads=config['num_heads'], num_classes=config['num_classes'], attn_dropout=config['attn_dropout'], 
        # dropout=config['dropout'], mlp_size=config['mlp_size'], positional_emb=config['positional_emb']).to(device)
        model = AttnSleep().to(device)
        
    
        # summary(model=model,
        #     input_size=(128, 1, 3000),
        #     col_names=["input_size", "output_size", "num_params", "trainable"],
        #     col_width=20,
        #     row_settings=["var_names"]
        # )
        
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = torch.optim.AdamW(trainable_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
        
        data_loader, valid_data_loader, test_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
                                                                    folds_data[fold_id][1], 
                                                                    folds_data[fold_id][2], batch_size)
        weights_for_each_class = calc_class_weight(data_count)

        print("Fold number is ", fold_id)
        
        # Test subject number
        valid_subj_number = os.path.split(folds_data[fold_id][1][0])[-1][3:5] #, os.path.split(folds_data[fold_id][1][1])[-1][3:5])
        test_subj_number = os.path.split(folds_data[fold_id][2][0])[-1][3:5]
        
        # print(next(iter(data_loader))[0].shape)
        print("Valid subject is", valid_subj_number)
        print("Test subject is", test_subj_number)
        
        # print(len(data_loader))
        # print(len(valid_data_loader))
        best_acc, best_f1 = 0, 0
        
        patience = 0
        max_patience = 15
        min_loss = 100

        train_accs = []
        train_losses = []
        valid_accs = []
        valid_losses = []

        for epoch in range(epochs): # epochs
            adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            train_acc, train_loss, valid_acc, valid_f1, valid_loss = train_epoch(model, epoch+1, epochs, 
                                            data_loader, optimizer, weighted_CrossEntropyLoss,
                                            weights_for_each_class, valid_data_loader)
            
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            valid_accs.append(valid_acc)
            valid_losses.append(valid_loss)
            
            if min_loss > valid_loss:
                min_loss = valid_loss
                patience = 0
                print(f"Saving the model on epoch {epoch}")
                torch.save(model.state_dict(), '{}/{}_best.pt'.format(folder_path, config_name)) 
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping on epoch {epoch}")
                    print(f"Last validation accuracy is {valid_acc}")
                    break
                
            best_acc, best_f1 = max(best_acc, valid_acc), max(best_f1, valid_f1)
            
            # print(f"Valid acc is {valid_acc}, valid_f1 is {valid_f1}")
        
        print('Loading best weights')
        model.load_state_dict(torch.load('{}/{}_best.pt'.format(folder_path, config_name)))
        test_acc, test_f1 = evaluate_model(model, test_data_loader, folder_path, config_name, test_subj_number)
        
        plot_training_results(train_accs, train_losses, valid_accs, valid_losses, config_name, test_subj_number, epoch+1, folder_path)
        
        results[test_subj_number] = test_acc
        print(results)
        
    save_to_csv(results, '{}/{}_test_results.csv'.format(folder_path, config_name))

    
if __name__ == '__main__':
    main()
    