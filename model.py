import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self , ctx_len , d_model):
        super().__init__()
                
        self.d_model = d_model
        self.pos = torch.arange(1 , ctx_len + 1).unsqueeze(1)
        self.i = torch.arange(0 , d_model / 2)
        self.R = torch.zeros(ctx_len , d_model , d_model)
        
    def forward(self , x):
        theta = 10000 ** (-2 * self.i) / self.d_model
        m_theta = self.pos * theta
        cos_values = torch.cos(m_theta)
        sin_values = torch.sin(m_theta)
                

class RMS_Norm(nn.Module):
    def __init__(self , d_model , eps = 1e-5):
        super().__init__()
        
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))
        
    def forward(self , x):
        norm = torch.sqrt(torch.mean(x**2 , dim = -1  , keepdim= True) + self.eps)
        rms_normed = (x / norm) * self.g
        return rms_normed

class FFN(nn.Module):
    def __init__(self , d_model , dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model , 4 * d_model , bias = False)
        self.fc2 = nn.Linear(4 * d_model , d_model , bias = False)
        self.fc3 = nn.Linear(d_model ,4 * d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self , x):
        x = self.fc2(self.dropout(F.silu(self.fc1(x) * self.fc3)))
        return x
    
