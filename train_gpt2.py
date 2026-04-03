from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import tiktoken
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024 # maximum sequence length
    vocab_size: int = 50257 # Number of tokens in the vocabulary: 50,000 BPE merges + 256 bytes tokens + 1 <end of text> token
    n_layer: int = 12 # Number of layers (Number of attention blocks)
    n_head: int = 6 # Number of attention heads
    n_embd: int = 768 # Embedding dimension

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
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # This copies the data pointer. The old value of wte.weight gets orphaned and garbage collected. 
        # The new value of wte.weight is the same as the new value of lm_head.weight.
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # Input is token indices of the input sentences.
        # idx is [B, T] where each element is an integer representing a token ID.
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # [T]
        pos_emb = self.transformer.wpe(pos) # [T, C]
        tok_emb = self.transformer.wte(idx) # [B, T, C]
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Flatting logits from (B, T, vocab_size) to (B * T, vocab_size)
            # Flatting targets from (B, T) to (B * T, )
            # This is because cross entropy doesn't like multi-dimensional targets.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 weights from Hugging Face."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embed are determined by the model type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create a from-scratch initialized minGPT model
        # Initialize config with hyperparameters
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # Discard this mask / buffer. The ".attn.bias" is just used for auto-regressive masking.

        # init a hunggiface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] # Ignore these

        # Some of the weights in the model we contructed are transposed when compared to the weight in huggingface
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(suffix) for suffix in transposed):
                # Special treatment for the Conv1D weights we need to transpose.
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters.
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T

        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y

num_return_sequences = 5
max_length = 30
device = "mps" if torch.backends.mps.is_available() else "cpu"

# enc = tiktoken.get_encoding("gpt2")
# with open('input.txt', 'r') as f:
#     text = f.read()
# tokens = enc.encode(text)
# B, T = 4, 32 # batch size, sequence length
# buf = torch.tensor(tokens[:B * T + 1])
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)
# x = x.to(device)
# y = y.to(device)

# get logits
# model = GPT.from_pretrained("gpt2")
model = GPT(GPTConfig())
model.to(device)
# logits, loss = model(x, y)

# AdamW is a bug fix of Adam. It is a more stable version of Adam.
# It is a more stable version of Adam.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train_loader = DataLoader(B = 4, T = 128)
for i in range(50):
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    # Always make sure to zero the gradients before backpropagation.
    optimizer.zero_grad()
    # Forward pass
    logits, loss = model(x, y)
    # Backward pass
    loss.backward()
    # Update the parameters
    optimizer.step()
    # Print the loss
    # loss.item() takes the 1D tensor, ship it back to the CPU.
    print(f"Step {i}, loss: {loss.item()}")


import sys; sys.exit(0)


# prefix tokens
model.eval()
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # [num_return_sequences, 8]
x = tokens.to(device)

torch.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # Tells that we don't need to compute the gradients
        logits, _ = model(x) # [num_return_sequences, x.size(1), vocab_size]

        # We only care about the logits for the last token
        logits = logits[:, -1, :] # [num_return_sequences, vocab_size]

        # get the probabilties
        probs = F.softmax(logits, dim=-1) # [num_return_sequences, vocab_size]

        # do top-k sampling of 50 (huggingface pipeline default)
        # tpok_probs here becomes (num_return_sequences, 50). topk_indices becomes (num_return_sequences, 50).
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)

        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1) # [num_return_sequences, 1]

        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # [num_return_sequences, 1]

        # append to the sequence
        x = torch.cat([x, xcol], dim=1) # [num_return_sequences, x.size(1) + 1]

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)