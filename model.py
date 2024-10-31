import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    def __init__(self):
        
        self.d_model = 4096
        self.vocab_size = 32000
        self.n_heads = 32
        self.n_layers = 32
        self.kv_heads = None 
        self.max_seq_len = 1024
        self.dropout = 0.1

def calc_rope_theta(config):

    d_model = config.d_model

    pos = torch.arange(config.max_seq_len)  
    i = torch.arange(0, d_model, 2)      

    theta = 1.0 / (10000 ** (i / d_model))  
    freq = torch.outer(pos, theta)            
    freqs_complex = torch.polar(torch.ones_like(freq), freq)
    
    return freqs_complex

def rotate(x: torch.Tensor, freqs_complex: torch.Tensor) -> torch.Tensor:

    assert x.shape[-1] % 2 == 0, "d_model must be even"
    
    x_complex = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_complex)
    
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    x_rotated = x_complex * freqs_complex
    
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    
    return x_out
                
class RMS_Norm(nn.Module):
    
    def __init__(self , config , eps = 1e-5):
        super().__init__()
        
        self.eps = eps
        self.g = nn.Parameter(torch.ones(config.d_model))
        
    def forward(self , x):
        
        norm = torch.rsqrt(torch.mean(x**2 , dim = -1  , keepdim= True) + self.eps)
        rms_normed = x * norm * self.g
        
        return rms_normed
    
class FFN(nn.Module):
    
    def __init__(self ,config):
        super().__init__()
        
        self.fc1 = nn.Linear(config.d_model , 4 * config.d_model , bias = False)
        self.fc2 = nn.Linear(4 * config.d_model , config.d_model , bias = False)
        self.fc3 = nn.Linear(config.d_model ,4 * config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self , x):
        
        x = self.fc1(x)
        return x
    
class MHA(nn.Module):
    
    def __init__(self , config):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.kv_heads = config.kv_heads
        self.rep = config.n_heads_q // self.kv_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // self.n_heads
        
        self.w_q = nn.Linear(config.d_model , self.n_heads * self.d_k , bias =  False)
        self.w_k = nn.Linear(config.d_model , self.kv_heads * self.d_k , bias = False)
        self.w_v = nn.Linear(config.d_model , self.kv_heads * self.d_k , bias = False)
        self.w_o = nn.Linear(self.n_heads * self.d_k , config.d_model  , bias = False)
        
        self.k_cache =  torch.zeros(config.max_batch_size , config.max_seq_len , config.kv_heads , config.d_model)
        self.v_cache =  torch.zeros(config.max_batch_size , config.max_seq_len , config.kv_heads , config.d_model)
            
    def forward(self , Q , K , V , x , rope_thetas):
        b , seq_len , _ = x.xhape
        
        Q = self.w_q(Q).view(self.n_heads , self.d_model)
        K = self.w_k(K).view(self.kv_heads , self.d_model)
        V = self.w_v(V).view(self.kv_heads , self.d_model)
        
        Q = Q.view(b , seq_len , self.n_heads , self.d_k)
        K = K.view(b , seq_len , self.kv_heads , self.d_k)
        V = V.view(b , seq_len , self.kv_heads , self.d_k)
        
        Q_rotated  = rotate(rope_thetas , Q)
        K_rotated = rotate(rope_thetas , K)
    
        attention_scores = torch.softmax((Q_rotated @ K_rotated) // (self.d_k) , dim = -1)
        attention_weights = attention_scores @ self.v_cache
        attention_out = self.w_o(attention_weights)
        return attention_out
        
class LLaMA_Block(nn.Module):
    
    def __init__(self , config):
        super().__init__()
        
        self.mha = MHA(config)
        self.ffn = FFN(config)
        
        self.norm_1 = RMS_Norm(config)
        self.norm_2 = RMS_Norm(config)

    def forward(self , x):
        
        norm_1 = self.norm_1(x)
        attention_out = self.mha(norm_1) + x
        
        norm_2 = self.norm_2(attention_out)
        ffn_out = self.ffn(norm_2) + attention_out
        
        return ffn_out

class LLaMA(nn.Module):
    
    def __init__(self , config):
        super().__init__()
        
        self.embeddings = nn.Embedding(config.vocab_size , config.d_model)
        self.rope_m_thetas = RoPE(config)
        
        self.layers = nn.ModuleList([LLaMA_Block(config) for _ in range(config.n_layers)])
        self.norm = RMS_Norm(config)
        self.linear = nn.Linear(config.d_model , config.vocab_size)
        
    def forward(self , x):
        
        embeddings = self.embeddings(x)
    
        
        for layer in self.layers(x):
            x = layer(x)
        
        x = self.norm(x)
        output = self.linear(x)
        
        return output

config = Config()
model = LLaMA(config)






        