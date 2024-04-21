import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math


def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    if "78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"
    else:
        r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
        # print(len(r_permute))
        # print(77 in r_permute)
    else:
        print ("============== ERROR =================")
        
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    
    # print(len(files_dict))        
    
    files_pairs = []
    for key in files_dict:
        # print(key)
        # break
        if len(files_dict[key]) == 2:
            files_pairs.append(files_dict[key])
        
    counter = 0
    counter_2 = 0
    for i in range(len(files_pairs)):
        if len(files_pairs[i]) == 1:
            counter += 1
        elif len(files_pairs[i]) == 2:
            counter_2 += 1
        # print(len(files_pairs[i]))
        
    print("Total number of 1s is", counter)
    print("Total number of 2s is", counter_2)
    
    mask = np.logical_or(r_permute < counter_2, r_permute >= 78)
    r_permute = r_permute[mask]
    
    print(len(r_permute))
    
    print(r_permute)
    
    # for el in range(counter_2, 78):
    #     r_permute.remove(el)
        
    files_pairs = np.array(files_pairs)
    files_pairs = files_pairs[r_permute]
    
    # for x in range(len(files_pairs)):
    #     print(os.path.split(files_pairs[x][0])[-1][3:5], os.path.split(files_pairs[x][1])[-1][3:5])

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight
