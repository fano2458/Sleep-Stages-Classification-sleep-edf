import torch
from torch.utils.data import Dataset

import os
import numpy as np


class LoadDataset_from_numpy(Dataset):
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()
        
        X_train = np.load(np_dataset[0])['x']
        y_train = np.load(np_dataset[0])['y']
        
        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)['x']))
            y_train = np.append(y_train, np.load(np_file)['y'])
            
        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
        
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
            else:
                self.x_data = self.x_data.unsqueeze(1)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    

def data_generator_np(training_files, subject_files, test_files, batch_size):
    train_dataset = LoadDataset_from_numpy(training_files)
    valid_dataset = LoadDataset_from_numpy(subject_files)
    test_dataset = LoadDataset_from_numpy(test_files)
    
    all_ys = np.concatenate((train_dataset.y_data, valid_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)
    
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader, counts
