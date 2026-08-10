import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

class BigramLanguageModel_FA(nn.Module):
    def __init__(self, vocab_size, n_embed=384, num_heads=6, block_size=256):
        super().__init__()
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #this maps each token to a vector of size "n_embed" #semantic meaning of the token is captured in the vector representation, which is learned during training
        self.position_embedding_table = nn.Embedding(block_size, n_embed) #T by C, maps the position of each token in the input sequence to a vector of size "n_embed" #this allows the model to capture the order of the tokens in the input sequence, which is important for language modeling
        # Use ModuleList so we can support passing per-block KV caches
        self.blocks = nn.ModuleList([
            Block(n_embed, num_heads, block_size),
            Block(n_embed, num_heads, block_size),
            Block(n_embed, num_heads, block_size),
            Block(n_embed, num_heads, block_size),
            Block(n_embed, num_heads, block_size),
        ]) #this is a stack of transformer blocks, each block contains a multi-head self-attention layer and a feed-forward layer
        self.ln_f = LayerNorm(n_embed) #final layer norm before the output layer
        self.lm_head = nn.Linear(n_embed, vocab_size) #the attention block projects back to the embedding size before predicting the next token
        #vector of size n_embed for each token in the input sequence, which is then used to predict the next token in the sequence
    def forward(self, idx, targets=None, past_key_values=None, use_cache=False): 
        """
        If `past_key_values` is provided it should be a list of length `len(self.blocks)` where
        each element is a tuple (past_k, past_v) with shapes (B, num_heads, T_past, head_size).
        When `use_cache` is True the model returns (logits, loss, present_key_values) where
        present_key_values has the same structure as `past_key_values` and contains the
        concatenated keys/values for each block.
        """
        B, T = idx.shape
        past_len = 0
        if past_key_values is not None:
            # past_key_values[i] = (past_k, past_v), and past_k shape is
            # (B, num_heads, T_past, head_size)
            past_len = past_key_values[0][0].size(2)
        token_embed = self.token_embedding_table(idx) # (B, T, n_embed)
        pos_ids = torch.arange(past_len, past_len + T, device=idx.device)
        pos_embed = self.position_embedding_table(pos_ids) # (T, n_embed)
        x = token_embed + pos_embed

        present_key_values = []
        # pass through each block, optionally using per-block past kv
        for i, block in enumerate(self.blocks):
            past = None
            if past_key_values is not None:
                past = past_key_values[i]
            x, present = block(x, past_key_value=past)
            if use_cache:
                present_key_values.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        if use_cache:
            return logits, loss, present_key_values
        return logits, loss

    def generate(self, idx, max_new_tokens): #idx is B,T
        """Autoregressive generation using KV cache. Returns generated indices (B, T+max_new_tokens)."""
        device = idx.device
        idx = idx[:, -self.block_size :]
        past_key_values = None
        for i in range(max_new_tokens):
            if i == 0:
                # prime the cache from the full context window
                idx_cond = idx.to(device)
            else:
                # then decode token-by-token using cache
                idx_cond = idx[:, -1:].to(device)

            logits, _loss, present = self(idx_cond, past_key_values=past_key_values, use_cache=True)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            past_key_values = present
        return idx
    

class Head(nn.Module):
    """one head of attention"""
    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix of shape (block_size, block_size). this ensures only the previous tokens can affect the curr token
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, past_k=None, past_v=None):
        """Compute attention for this head. If past_k/past_v are provided they will be
        concatenated to the current keys/values and returned as the present kv for caching.
        Returns: out, (present_k, present_v)
        """
        B, T, C = x.shape
        k_cur = self.key(x)   # (B, T, head_size)
        q = self.query(x)     # (B, T, head_size)
        v_cur = self.value(x) # (B, T, head_size)

        if past_k is None:
            k = k_cur
            v = v_cur
            is_causal = True
        else:
            # past_k / past_v expected shapes: (B, T_past, head_size)
            k = torch.cat([past_k, k_cur], dim=1)
            v = torch.cat([past_v, v_cur], dim=1)
            if k.size(1) > self.tril.size(0):
                k = k[:, -self.tril.size(0) :, :]
                v = v[:, -self.tril.size(0) :, :]
            is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        return out, (k, v)
    
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, n_embed, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed) #this is the O proj
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, past_key_value=None):
        """If `past_key_value` is provided it should be a tuple (past_k, past_v) with
        shapes (B, num_heads, T_past, head_size). Returns: out, (present_k, present_v)
        where present_k/present_v have shapes (B, num_heads, T_total, head_size).
        """
        # collect outputs and present kvs per head
        outs = []
        present_ks = []
        present_vs = []

        if past_key_value is None:
            # simple path: no past
            for h in self.heads:
                out_h, (k_h, v_h) = h(x)
                outs.append(out_h)
                present_ks.append(k_h)
                present_vs.append(v_h)
        else:
            past_k, past_v = past_key_value
            # past_k/past_v: (B, num_heads, T_past, head_size)
            for i, h in enumerate(self.heads):
                # slice per-head past
                pk = past_k[:, i]  # (B, T_past, head_size)
                pv = past_v[:, i]
                out_h, (k_h, v_h) = h(x, past_k=pk, past_v=pv)
                outs.append(out_h)
                present_ks.append(k_h)
                present_vs.append(v_h)

        # concat heads' output on last dim
        out = torch.cat(outs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        # stack present kvs: list of (B, T_total, head_size) -> (B, num_heads, T_total, head_size)
        present_k = torch.stack(present_ks, dim=1)
        present_v = torch.stack(present_vs, dim=1)
        return out, (present_k, present_v)
    
class FeedForward(nn.Module):
    """linear layer followed by a non-lienar activation function"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, num_heads, block_size):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(num_heads=num_heads, n_embed=n_embed, head_size=head_size, block_size=block_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)
    def forward(self, x, past_key_value=None):
        """Returns (x, present_key_value) where present_key_value is (present_k, present_v)
        for this block. past_key_value should be None or a tuple (past_k, past_v) with
        shapes (B, num_heads, T_past, head_size).
        """
        sa_out, present = self.sa_head(self.ln1(x), past_key_value=past_key_value)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, present

class LayerNorm(nn.Module):
    """Layer normalization"""
    def __init__(self, n_embed, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased = False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
