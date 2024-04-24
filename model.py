import numpy as np
import torch
import torch.nn as nn

## While doing layernorm, x is in shape (batch_size, seq_len, embedding_dim) but attention output is in shape (batch_size, seq_len, d_model)
## This causes error. SO I changed attention output to embedding dim

## self.mask[:T, :T] --> X is of shape (B, context_len, EMbedding_Dim) but that is only true while training. 
## We do give smaller context length while testing or generating text
## So, we need to make sure that the model is able to handle that as well. Mask will be used for that purpose [:T, :T]

## Still facing issue with positional encoding. --> somehow, the positional encoding tensors are on cpu and everything else is on gpu
## For now I am moving it on the gpt here, but ideally I would want to do it by calling Model object and doing obj.to(device)

def positional_encoding(context_length, embedding_dim):
    pos = np.arange(context_length).reshape(-1, 1)
    i = np.arange(embedding_dim).reshape(1, -1)
    angle = pos / np.power(10000, 2 * i / embedding_dim)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.Tensor(angle).to(device)

class attention_head(nn.Module):
    def __init__(self, d_model, dropout = 0.1, embedding_dim = 128, context_length = 128):
        super().__init__()

        self.d_model = d_model
        self.K = nn.Linear(in_features = embedding_dim, out_features = d_model, bias=False)
        self.Q = nn.Linear(in_features = embedding_dim, out_features = d_model, bias=False)
        self.V = nn.Linear(in_features = embedding_dim, out_features = d_model, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        attention_scores = Q @ K.transpose(-2, -1) ## Q -> (B, L, d_model), K -> (B, d_model, L) => K.transpose(-2, -1) -> (B, L, d_model) => Q @ K.transpose(-2, -1) -> (B, L, L)
        attention_scores /= np.sqrt(self.d_model)
        attention_scores = attention_scores.masked_fill(self.mask[:T, :T] == 0, -1e9)
        attention_scores = self.softmax(attention_scores) ## (B, L, L) -> (B, L, L)
        attention_scores = self.dropout(attention_scores)
        attention = attention_scores @ V ## (B, L, L) @ (B, L, d_model) -> (B, L, d_model)

        return attention ## (B, L, d_model)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout = 0.1, embedding_dim = 128, context_length = 128):
        super().__init__()

        self.heads = nn.ModuleList([attention_head(d_model = d_model, dropout = dropout, embedding_dim = embedding_dim, context_length = context_length) for _ in range(num_heads)])
        self.linear = nn.Linear(in_features = num_heads * d_model, out_features = embedding_dim, bias = False)

    def forward(self, x):
        ## x -> (B, L, EMBEDDING_DIM)
        attention = torch.cat([head(x) for head in self.heads], dim = -1) ## (B, L, num_heads * d_model)
        return self.linear(attention) ## (B, L, d_model)

class DecoderBlock(nn.Module):
    def __init__(self, num_heads, d_model, dropout = 0.1, context_length = 128, embedding_dim = 128):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(num_heads = num_heads, d_model = d_model, dropout = dropout, embedding_dim = embedding_dim, context_length = context_length) 
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features = embedding_dim, out_features = 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features = 4 * embedding_dim, out_features = embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        ## x -> (B, L, EMBEDDING_DIM)
        attention = self.multi_head_attention(x) ## (B, L, EMBEDDING_DIM)
        x = self.norm1(x + attention) ## (B, L, EMBEDDING_DIM)
        feed_forward = self.feed_forward(x) ## (B, L, EMBEDDING_DIM)
        return self.norm2(x + feed_forward) ## (B, L, EMBEDDING_DIM)

class Decoder_layer(nn.Module):
    def __init__(self, num_heads, d_model, num_layers = 6, dropout = 0.1, context_length = 128, embedding_dim = 128):
        super().__init__()

        self.blocks = nn.ModuleList([DecoderBlock(num_heads = num_heads, d_model = d_model, dropout = dropout, context_length = context_length, embedding_dim = embedding_dim) for _ in range(num_layers)])

    def forward(self, x):
        ## x -> (B, L, EMBEDDING_DIM)
        for block in self.blocks:
            x = block(x)
        return x ## (B, L, d_model)
    
class Model(nn.Module):
    def __init__(self, num_heads, d_model, vocab_size, dropout = 0.1, context_length = 128, embedding_dim = 128, num_layers = 6, padding_idx = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.lang_embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim, padding_idx=padding_idx)
        self.positional_encoding = positional_encoding(context_length, embedding_dim)
        self.decoder = Decoder_layer(num_heads = num_heads, d_model = d_model, num_layers = num_layers, dropout = dropout, context_length = context_length, embedding_dim = embedding_dim) 
        self.linear = nn.Linear(in_features = embedding_dim, out_features = vocab_size, bias = False)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        ## x -> (B, L)
        x = self.lang_embedding(x) ## (B, L, EMBEDDING_DIM)
        x += self.positional_encoding[:x.shape[1], :].unsqueeze(0) ## pos encoding is in (L, EMBEDDING_DIM)--> unsqueeze turns into (1, L, EMBEDDING_DIM) --> addition broadcasts into (B, L, EMBEDDING_DIM)
        x = self.decoder(x) ## (B, L, d_model)
        x = self.linear(x) ## (B, L, vocab_size)
        return self.softmax(x) ## (B, L, vocab_size)