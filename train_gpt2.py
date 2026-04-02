from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dimension.
        # nh = number of heads
        # hs = head size
        # C = number of channels  = nh * hs
        # In GPT-2 (124M), n_head = 12, hs = 64, so, nh * hs = 768 channels in the transformer.
        # Each token emits 3 vectors: key, query, value.
        # When you multiple a query and a key, you get the attention amount.
        # qkv is [B, T, 3 * C]
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # q is [B, T, n_head, n_embd]
        # k is [B, T, n_head, n_embd]
        # v is [B, T, n_head, n_embd]

        # We are making the number of heads into the batch dimension.
        # In the operations that follow, PyTorch treats the first two dimensions as the batch dimensions.
        # It applies operations to all of them in parallel, within both the batch and the within the heads.
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, hs]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, hs]

        # attention materializes the large (T,T) matrix for all the queries and keys.
        attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))

        # Apply the bias: :: means all indices from start to end
        # This is to make sure the tokens only attend to previous tokens and never to the tokens in the future.
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # softmax is a weight sum function.
        # Normalizes the attention scores to sum to 1.
        attn = F.softmax(attn, dim=-1)

        # Attention matrix mulitply with the values. 
        # This is a weighted sum of the values, weighted by the attention scores.
        y = attn @ v # [B, n_head, T, T] * [B, n_head, T, hs] = [B, n_head, T, hs]

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Reassable all the head output side by side. [B, n_head, T, hs] -> [B, T, n_head * hs]
        
        y = self.c_proj(y)

        return y

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Addition distributes gradients to both of its branches equally during backpropagation.
        # So, the gradient here flows directly to the inputs through the residual connections unchanged.
        # Clean residual pathway is desirable from an optimization perspective.

        # Attention is a communication operation.
        # It is where the tokens communicate with each other and exchange information.
        # It is a weight sum function. It is a reduce operation.
        x = x + self.attn(self.ln_1(x))

        # MLP is done for each token independently. There is no communication between tokens.
        # There is no information exchanged between the tokens.
        # This is where they think individually about the information they gathered.
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ModuleDict will allow you to index into the submodules like keys in a dictionary.
        self.transformer = nn.ModuleDict(dict(
            # nn.Embed is just a wrapper around nn.Linear that initializes the weights to random values.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # ModuleList will allow you to iterate over the submodules like a list.
            # We can index into the submodules like a list 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.llm_head = nn.Linear(config.n_embd, config.vocab_size)
