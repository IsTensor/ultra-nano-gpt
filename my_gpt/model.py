from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # for linear and Layerform

class CausalSelfAttention(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 3. Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 4. Store hyperparameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        
        # 5. Flash Attention Mask
        # This is a lower triangular matrix to ensure the model cannot "see the future"
        # register_buffer means it's not a trainable parameter but part of the state_dict
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self,x):
        # x shape: (Batch_Size, Time_Step, Channel/Embed_Dim) = (B, T, C)
        B,T,C = x.size()
        # qkv = (B,T,3C)
        qkv= self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)

        # split into multi head
        # q shape (B,T,C) -> target : (B,T, n_head, head_dim)
        # for batch calculation, we permute the last two dimensions

        q = q.view(B,T,self.n_head,self.head_dim).permute(0,2,1,3)
        k = k.view(B,T,self.n_head,self.head_dim).permute(0,2,1,3)
        v = v.view(B,T,self.n_head,self.head_dim).permute(0,2,1,3)
        
        # Using F.scaled_dot_product_attention (Flash Attention if available)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        # shape of y : (B, n_head, T, head_dim)
        
        # reshape and concat y
        y = y.permute(0,2,1,3).contiguous().view(B,T,C)

        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd, bias = config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        # x shape (B, T, n_embd)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module): # Transformer Block
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd,bias = config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd,bias = config.bias)

    def forward(self,x):
        # residual of attention
        x = x + self.attn(self.ln_1(x))
        # residual of MLP
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights:
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        self.block_size = config.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # idx is the index of token, shape:(Batch_size,time_steps)
        device = idx.device
        b, t = idx.shape  # batch_size, time_steps
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device) # (t,)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x) # note: using x[:, [-1], :] might be more efficient but keeping it full seq for now to match interface
            loss = None
            
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type='gpt2'):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading official {model_type} weights...")

        # 1. Create the model to be initialized
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 
        config_args['block_size'] = 1024  
        config_args['bias'] = True        
        
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        # 2. Load HuggingFace official model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 3. Copy parameters
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for k in sd_keys_hf:
            k_my = k
            if k_my not in sd:
                # Debug print removed to keep it clean, but could be added back
                continue
            
            with torch.no_grad():
                if any(k.endswith(w) for w in transposed):
                    # Transpose Conv1D weights
                    assert sd_hf[k].shape[::-1] == sd[k_my].shape
                    sd[k_my].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[k_my].shape
                    sd[k_my].copy_(sd_hf[k])

        print("Weights loaded successfully!")
        return model
