import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    def __init__(self):
        
        self.d_model = 4096
        self.vocab_size = 32000
        self.n_heads = 12
        self.kv_heads = 6 
        self.n_layers = 12
        self.max_seq_len = 204
        self.dropout = 0.1

class RoPE(nn.Module):
    
    def __init__(self , config):
        super().__init__()
                
        self.d_model = config.d_model
        self.pos = torch.arange(1 , config.ctx_len + 1).unsqueeze(1)
        self.i = torch.arange(0 , config.d_model / 2)
        self.R = torch.zeros(config.ctx_len , config.d_model , config.d_model)
        
    def forward(self , x):
        
        theta = 10000 ** (-2 * self.i) / self.d_model
        m_theta = self.pos * theta
        
        cos_values = torch.cos(m_theta)
        sin_values = torch.sin(m_theta)
                
class RMS_Norm(nn.Module):
    
    def __init__(self , config , eps = 1e-5):
        super().__init__()
        
        self.eps = eps
        self.g = nn.Parameter(torch.ones(config.d_model))
        
    def forward(self , x):
        
        norm = torch.sqrt(torch.mean(x**2 , dim = -1  , keepdim= True) + self.eps)
        rms_normed = (x / norm) * self.g
        
        return rms_normed

class FFN(nn.Module):
    
    def __init__(self ,config):
        super().__init__()
        
        self.fc1 = nn.Linear(config.d_model , 4 * config.d_model , bias = False)
        self.fc2 = nn.Linear(4 * config.d_model , config.d_model , bias = False)
        self.fc3 = nn.Linear(config.d_model ,4 * config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self , x):
        
        x = self.fc2(self.dropout(F.silu(self.fc1(x) * self.fc3)))
        return x
    
class MHA(nn.Module):
    
    def __init__(self , config):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.kv_heads = config.kv_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // self.n_heads
        
        self.w_q = nn.Linear(config.d_model , config.d_model)
        self.w_k = nn.Linear(config.d_model , config.d_model)
        self.w_v = nn.Linear(config.d_model , config.d_model)
        
    def forward(self , Q , K , V):
        
        Q = self.w_q(Q).view(self.n_heads , self.d_model)
        K = self.w_k(K).view(self.kv_heads , self.d_model)
        V = self.w_v(V).view(self.kv_heads , self.d_model)
        
class LLaMABlock(nn.Module):
    
    def __init__(self , config):
        super().__init__()
        
        self.embeddings = nn.Embedding(config.vocab_size , config.d_model)
        self.rope_embeddings = RoPE(config)
        self.mha = MHA(config)
        self.ffn = FFN(config)
        
        self.norm_1 = RMS_Norm(config)
        self.norm_2 = RMS_Norm(config)

    def forward(self , x):
        
        embeddings = self.embeddings(x)
        norm_1 = self.norm_1(embeddings)
        attention_out = self.mha(norm_1) + x
        
        norm_2 = self.norm_2(attention_out)
        ffn_out = self.ffn(norm_2) + attention_out
        
        return ffn_out

class LLaMA(nn.Module):
    
    def __init__(self , config):
        super().__init__()
        
        self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(config.n_layers)])
        self.norm = RMS_Norm(config)
        self.linear = nn.Linear(config.d_model , config.vocab_size)
        
    def forward(self , x):
        
        for layer in self.layers(x):
            x = layer(x)
        
        x = self.norm(x)
        output = self.linear(x)
        
        return output
        

        