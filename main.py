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
    return valid_acc, valid_f1, valid_loss


def evaluate_model(model, test_data_loader):
    model.eval()
    
    # Initialize accumulators for test accuracy, loss and f1 score
    test_acc, test_f1 = 0, 0
    
    with torch.no_grad():
        
        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            acc = accuracy(output, target)
            f1_score = f1(output, target)
            
            # Accumulate the accuracy, loss and f1 score for each batch
            test_acc += acc
            test_f1 += f1_score
            
    # Calculate average accuracy, loss and f1 score over all test batches
    test_acc /= len(test_data_loader)
    test_f1 /= len(test_data_loader)
    
    return test_acc, test_f1
    

import csv

def save_to_csv(data, filename):
  with open(filename, 'w', newline='') as csvfile:
    fieldnames = ['test_sub_num', 'accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for subject_num, accuracy in data.items():
      writer.writerow({'test_sub_num': subject_num, 'accuracy': accuracy})


def main():
    # load config parameters
    config_name = 'cct_config'
    with open(config_name+'.json', 'r') as jsonfile:
        config = json.load(jsonfile)
    
    
    # Define the directory path
    folder_path = "checkpoints/{}_results/".format(config_name)

    try:
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    
    # criterion = weighted_CrossEntropyLoss
    metrics = [accuracy, f1]
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    batch_size = 128
    epochs = 150
    
    folds_data = load_folds_data(np_data_dir, 20)
    
    # print(config['kernel_sizes'])
    
    results = dict()
    
    for fold_id in tqdm(range(20)):        
        model = CCT(kernel_sizes=config['kernel_sizes'], stride=config['stride'], padding=config['padding'],
        pooling_kernel_size=config['pooling_kernel_size'], pooling_stride=config['pooling_stride'], 
        pooling_padding=config['pooling_padding'], n_conv_layers=config['n_conv_layers'], 
        n_input_channels=config['n_input_channels'], in_planes=config['in_planes'], activation=config['activation'], # ReLU
        max_pool=config['max_pool'], conv_bias=config['conv_bias'], dim=config['dim'], num_layers=config['num_layers'],
        num_heads=config['num_heads'], num_classes=config['num_classes'], attn_dropout=config['attn_dropout'], 
        dropout=config['dropout'], mlp_size=config['mlp_size'], positional_emb=config['positional_emb']).to(device)
    
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

        for epoch in range(150): # epochs
            adjust_learning_rate(optimizer=optimizer, epoch=epoch)
            valid_acc, valid_f1, valid_loss = train_epoch(model, epoch+1, epochs, 
                                            data_loader, optimizer, weighted_CrossEntropyLoss,
                                            weights_for_each_class, valid_data_loader)
            
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
        test_acc, test_f1 = evaluate_model(model, test_data_loader)
        
        results[test_subj_number] = test_acc
        print(results)
        
    save_to_csv(results, '{}/{}_test_results.csv'.format(folder_path, config_name))

    
if __name__ == '__main__':
    main()
    