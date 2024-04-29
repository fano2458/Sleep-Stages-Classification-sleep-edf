import torch
import torch.nn as nn
from .utils import Conv1d, MaxPool1d
from .model import Transformer

# sleepcopy
class DeepSleepNetFeature(nn.Module):
    def __init__(self):
        super(DeepSleepNetFeature, self).__init__()

        self.chn = 64

        # architecture
        self.dropout = nn.Dropout(p=0.5)
        self.path1 = nn.Sequential(Conv1d(1, self.chn, 50, 6, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(8, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME')
                                   )
        self.path2 = nn.Sequential(Conv1d(1, self.chn, 400, 50, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(2, padding='SAME'))

        self.compress = nn.Conv1d(self.chn*4, 128, 1, 1, 0)
        self.smooth = nn.Conv1d(128, 128, 3, 1, 1)
        self.conv_c5 = nn.Conv1d(128, 128, 1, 1, 0)
    
    def sequence_length(self, n_channels=1, height=1, width=3000): # MUST be changed
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
        
    def forward(self, x):
        out = []
        x1 = self.path1(x)  # path 1
        x2 = self.path2(x)  # path 2
        
        x2 = torch.nn.functional.interpolate(x2, x1.size(2))
        c5 = self.smooth(self.compress(torch.cat([x1, x2], dim=1)))
        
        p5 = self.conv_c5(c5)
        out.append(p5)

        return out
    
    
class CNN_Transformer(nn.Module):
    def __init__(self,
            dim, num_layers,
            num_heads, num_classes, 
            attn_dropout, dropout, 
            mlp_size, positional_emb,
            activation=None):
        super(CNN_Transformer, self).__init__()
        
        self.tokenizer = DeepSleepNetFeature()
        
        self.transformer = Transformer(
            dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes, 
            attn_dropout=attn_dropout, dropout=dropout, 
            mlp_size=mlp_size, positional_embedding=positional_emb, 
            sequence_length=128
        )
        
    def forward(self, x):
        # print("Init input")
        # print(x.shape)
        # x = x.unsqueeze(1)
        # print(x.shape)
        x = self.tokenizer(x)[0]
        # print(x[0].shape)
        x = self.transformer(x)
        return x
