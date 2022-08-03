import os, sys
import time, glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

sys.path.append('../../JKLib')
from torchJK import CosineAnnealingWarmUpRestarts, PositionalEncoding
# efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7, efficientnet-b8, efficientnet-l2
class IonNet(torch.nn.Module):
    def __init__(self, 
                 seq_len: int = 1,                 
                 in_channels: int = 1,
                 n_parameters: int = 7,      
                 emb_size: int = 512,
                 d_model: int = 512,           
                 dropout: float = 0.1
                ):
        super(IonNet, self).__init__()

        self.name = 'model3'
        self.seq_len = seq_len
        self.out_size = n_parameters
        self.emb_size = emb_size
        self.d_model = d_model

        self.emb = torch.nn.Linear(in_channels, emb_size)

        self.conv1 = torch.nn.Conv1d(in_channels=emb_size, out_channels=64, kernel_size=3, 
                  	stride=2, padding=0, dilation=2, 
                  	groups=1, bias=True, padding_mode='zeros')        
        self.conv1_bn = torch.nn.BatchNorm1d(64)
        self.relu1 = torch.nn.ReLU(inplace=False)

        self.eff = EfficientNet.from_pretrained('efficientnet-b4')
        self.eff._fc = torch.nn.Linear(self.eff._fc.in_features, n_parameters, bias=True)  

    def forward(self, src: torch.Tensor): # src : (B, S, F)   
        
        emb = self.emb(src)       # (B, S, emb_size)      
        out = emb.permute(0,2,1)  # (B, emb_size, S)        

        out = self.conv1(out)     
        out = self.conv1_bn(out)
        out = self.relu1(out)

        out = out.view(-1, 1, out.size(1), out.size(2))        
        out = out.repeat(1, 3, 1, 1)
        # print(out.size())
        out = self.eff(out)  # (B, d_model*n_parameters)      
                
        return out, out, out