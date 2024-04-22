import torch
import torch.nn as nn
import numpy as np

import json
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
    return valid_acc, valid_f1


def main():
    
    model = CCT(kernel_sizes=[(1, 25), (1, 25)], stride=(1, 1), padding=(0, 0),
            pooling_kernel_size=(1, 5), pooling_stride=(1, 5), pooling_padding=(0, 0),
            n_conv_layers=2, n_input_channels=1,
            in_planes=32, activation=nn.ReLU, # ReLU
            max_pool=True, conv_bias=False,    
            dim=32, num_layers=3,
            num_heads=4, num_classes=5, 
            attn_dropout=0.3, dropout=0.3, 
            mlp_size=64, positional_emb="learnable").to(device)
    
    summary(model=model,
        input_size=(32, 1, 3000),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )

    # criterion = weighted_CrossEntropyLoss
    metrics = [accuracy, f1]
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.AdamW(trainable_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    batch_size = 32
    epochs = 100
    
    folds_data = load_folds_data(np_data_dir, 19)
    
    for fold_id in range(19):
        data_loader, valid_data_loader, data_count = data_generator_np(folds_data[fold_id][0],
                                                                    folds_data[fold_id][1], batch_size)
        weights_for_each_class = calc_class_weight(data_count)

        print("Fold number is ", fold_id)
        
        # Test subject number
        test_subj_number = os.path.split(folds_data[fold_id][1][0])[-1][3:5] #, os.path.split(folds_data[fold_id][1][1])[-1][3:5])
        # print(next(iter(data_loader))[0].shape)
        print("Test subject is", test_subj_number)
        
        # print(len(data_loader))
        # print(len(valid_data_loader))

        for epoch in tqdm(range(epochs)):
            valid_acc, valid_f1 = train_epoch(model, epoch+1, epochs, 
                                            data_loader, optimizer, weighted_CrossEntropyLoss,
                                            weights_for_each_class, valid_data_loader)
            
            # print(f"Valid acc is {valid_acc}, valid_f1 is {valid_f1}")
        
    
if __name__ == '__main__':
    main()
    