import torch
import torch.nn as nn
import numpy as np

import json
from data_loader.data_loaders import *


seed = 2024
torch.manual_seed(seed=seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(seed=seed)


with open('config.json', 'r') as f:
    config = json.load(f)


def main():
    batch_size = config['data_loader']['args']['batch_size']
    print(batch_size)



if __name__ == '__main__':
    main()
    