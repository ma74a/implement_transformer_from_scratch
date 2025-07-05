import torch
import torch.nn as nn

import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    # x -> [batch_size, seq_len]
    def forward(self, x):
        # We multiply the embedding with sqrt of the d_model
        return self.embedding(x) * math.sqrt(d_model) # [batch_size, sequence_length, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        pe = torch.zeros(seq_len, d_model) # [seq_len, d_model]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position*div_term) # [seq_len, d_model]
        pe[:, 1::2] = torch.cos(position*div_term) # [seq_len, d_model]
        
        pe = pe.unsqueeze(0) # [1, seq_len, d_model] for the batch
        self.register_buffer("pe", pe)
    
    # x -> [batch_size, seq_len, d_model]
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # check if the d_model is divisable by num_heads or NOT
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # the d_model of each head to Q,K,V
        
        self.q_w = nn.Linear(in_features=d_model, out_features=d_model) # Query [d_model, d_model]
        self.q_k = nn.Linear(in_features=d_model, out_features=d_model) # Key
        self.q_v = nn.Linear(in_features=d_model, out_features=d_model) # Value
        self.q_o = nn.Linear(in_features=d_model, out_features=d_model) # Output
    
    # x -> [batch_size, seq_len, d_model]
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k) # [batch_size, seq_len, num_head, d_k]
        x = x.transpose(1, 2) # [batch_size, num_head, seq_len, d_k]
        return x
    
    def scaled_dot_product_attention(self, Q, K, V, mask):
        # We make transpose to K to match the dim to Q:
        # Q = K = [batch_size, num_head, seq_len, d_k]
        # K.transpose(-2, -1) -> [batch_size, num_head, d_k, seq_len]
        # the inner dim should be equal (d_k, d_k)
        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask to prevent looking to the future
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        atten_probs = torch.softmax(atten_scores, dim=-1)
        
        atten_output = torch.matmul(atten_probs, V) # [batch_size, num_heads, seq_len, d_k]
        return atten_output 
    
    # x -> [batch_size, num_heads, seq_len, d_k]
    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(2, 1).contiguous().view(batch_size, seq_len, self.d_model) # [batch_size, seq_len, d_model]
        return x
    
    # Q = K = V -> [batch_size, seq_len, d_model]
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.q_w(Q))
        K = self.split_heads(self.K_w(K))
        V = self.split_heads(self.q_w(V))
        
        atten_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        combined_heads = self.combine_heads(atten_output)
        
        outputs = self.q_o(combined_heads) # [batch_size, seq_len, d_model]
        return outputs

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        # d_ff: Dimensionality of the inner layer in the feed-forward network.
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float=10**6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(d_model)) # Added
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        

    
    
      

vocab_size = 50
d_model = 512
q = torch.randn(vocab_size, d_model).unsqueeze(0)
k = torch.randn(vocab_size, d_model).unsqueeze(0)
v = torch.randn(vocab_size, d_model).unsqueeze(0)
# print(q.shape) # [batch_size, seq_len, d_model]
batch_size, seq_len, d_model = q.size()
q = q.view(batch_size, seq_len, 4, 128)
k = k.view(k.size(0), k.size(1), 4, 128)
# print(q.shape)
q = q.transpose(1, 2)
k = k.transpose(1, 2)
# print(q.shape)
# print(k.shape)
o = torch.matmul(q, k.transpose(-2, -1))
print(o.shape)
o = torch.matmul(o, v)
print(o.shape)
# x = o.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
# print(x.shape)
# print(k.transpose(-2, -1).shape)




# inputs = torch.randint(0, vocab_size, (2, 10))
# print(inputs.shape)
# i = InputEmbeddings(vocab_size, d_model)
# print(i(inputs).shape)
# pe = torch.zeros(vocab_size, d_model)
# print(pe.shape)
# position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
# # print(position)
# print(position.shape)
# div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
# # print(div_term)
# print(div_term.shape)
# pe[:, 0::2] = torch.sin(position*div_term)
# print(pe.shape)
# pe = pe.unsqueeze(0)
# print(pe.shape)
# p = pe[:,:pe]