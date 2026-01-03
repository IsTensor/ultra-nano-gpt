from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # 3. 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 4. 存储一些超参数
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        
        # 5. Flash Attention 掩码 (Mask)
        # 这是一个下三角矩阵，用来保证模型"看不见未来"
        # 也就是：第 T 个词只能看到 0...T 的词，看不到 T+1...
        # register_buffer 意味着它不是可训练参数，但在保存模型时会被存下来
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self,x):
        # x 的形状: (Batch_Size, Time_Step, Channel/Embed_Dim) = (B, T, C)
        B,T,C = x.size()
        # X (B,T,C)  c_attn(n_embd, 3*n_embd)   -> B,T,3C (c = n_embd)
        # qkv = (B,T,3C)
        qkv= self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)

        #split into multi head
        # q shape (B,T,C) -> target : (B,T, n_head, head_dim)
        #  for batch calculation, we permute the last two dimensions

        q = q.view(B,T,self.n_head,self.head_dim).permute(0,2,1,3)
        k = k.view(B,T,self.n_head,self.head_dim).permute(0,2,1,3)
        v = v.view(B,T,self.n_head,self.head_dim).permute(0,2,1,3)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        # shape of y : (B, n_head, T, head_dim)
        # reshape and concate y
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

class Block(nn.Module): #Transformer Block
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
        self.apply(self._init_weights)
        self.block_size = config.block_size

    def _init_weights(self, module):
        # 简单的初始化逻辑，符合 GPT-2 标准
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
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
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type='gpt2'):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"正在加载官方 {model_type} 权重...")

        # 1. 创建我们要初始化的模型
        # 对应不同型号的配置
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # GPT-2 总是这个大小
        config_args['block_size'] = 1024  # GPT-2 总是 1024
        config_args['bias'] = True        # GPT-2 总是使用 bias
        
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # 过滤掉一些不需要的 key (比如 mask buffer)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        # 2. 加载 HuggingFace 的官方模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 3. 开始搬运参数
        # 我们需要处理 naming 不一致的问题
        # 你的代码里 Block 里的 MLP 是 self.MLP，官方是 self.mlp
        # 你的代码里 LayerNorm 是 ln_1, ln_2，官方是 ln_1, ln_2 (一致)
        # 你的代码里 Attention 是 attn，官方是 attn (一致)
        
        sd_keys_hf = sd_hf.keys()
        # 过滤 HF 不需要的部分
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] 
        
        # 这里的关键是：Conv1D 的权重需要转置
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 考虑到你的命名可能是 MLP，我们需要做一个映射检查
        
        assert len(sd_keys_hf) == len(sd_keys), f"参数数量不匹配! HF: {len(sd_keys_hf)}, Yours: {len(sd_keys)}"

        for k in sd_keys_hf:
            # 尝试把 HF 的 key 映射到你的 key
            # 主要是处理 MLP vs mlp 的大小写问题

            k_my = k

            if k_my not in sd:
                print(f"警告: 在你的模型里找不到 key: {k_my} (原始: {k})")
                continue
            
            # 复制参数
            with torch.no_grad():
                # 判断是否需要转置
                # HF 的 Conv1D 权重形状是 [in, out]，我们需要 [out, in]
                if any(k.endswith(w) for w in transposed):
                    # 特殊处理：Conv1D 权重需要转置
                    assert sd_hf[k].shape[::-1] == sd[k_my].shape
                    sd[k_my].copy_(sd_hf[k].t())
                else:
                    # 普通参数直接复制
                    assert sd_hf[k].shape == sd[k_my].shape
                    sd[k_my].copy_(sd_hf[k])

        print("权重加载完毕！")
        return model

# after having the model, we do some simple shakespear training
import time

# 1. 读取文本数据
# 为了方便，你可以直接复制这段文本，或者去下载 tiny_shakespeare.txt
# 这里我们用一个简短的 demo 文本，实际训练建议下载: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. 构建字符级 Tokenizer
chars = sorted(list(set(text))) # 找出所有出现的字符
vocab_size = len(chars)
print(f"总字符数: {len(text)}")
print(f"词表大小 (vocab_size): {vocab_size}")
print(f"词表: {''.join(chars)}")

# 字符 -> 数字
stoi = { ch:i for i,ch in enumerate(chars) }
# 数字 -> 字符
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 编码: string -> list[int]
print(len(encode(text)))
decode = lambda l: ''.join([itos[i] for i in l]) # 解码: list[int] -> string

# 3. 把整个数据集编码成 Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# 90% 用于训练，10% 用于验证
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# -----------------------------------------------------------------------------
# 训练辅助函数
# -----------------------------------------------------------------------------
def get_batch(split, block_size, batch_size, device):
    # 根据是训练还是验证，选择数据源
    data = train_data if split == 'train' else val_data
    
    # 随机生成 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 提取 x (输入) 和 y (目标)
    # stack 将多个 tensor 堆叠起来
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # 移动到 GPU/CPU
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # 评估时不需要算梯度
def estimate_loss(model, eval_iters, block_size, batch_size, device):
    out = {}
    model.eval() # 切换到评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, device)
            logits = model(X)
            # 计算 Loss (需要把 logits 展平)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 切回训练模式
    return out


if __name__ == "__main__":
    import torch
    from torch.nn import functional as F

    # 1. 自动加载官方权重
    # 这会下载约 500MB 的文件
    model = GPT.from_pretrained('gpt2')
    
    # 2. 放到 GPU 上 (如果可用)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # 3. 准备输入
    # 我们用 tiktoken 或者简单的 encode (如果你装了 tiktoken 最好，没装也没事)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    except ImportError:
        print("未安装 tiktoken，无法精确测试 GPT-2 的生成能力（因为词表不同）。")
        print("请运行: pip install tiktoken")
        exit()

    # 给它一个提示
    # prompt = "Alan Turing is defined as"
    prompt = "Today I want to have something for lunch, and here is my choice: "
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...] # (1, T)

    # 4. 生成
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    
    with torch.no_grad():
        # 生成 50 个 token
        for k in range(50):
            logits = model(x)
            # 取最后一个时间步
            logits = logits[:, -1, :]
            # 简单的 Greedy Search (取概率最大的)，方便看有没有乱码
            # 也可以用 multinomial 采样
            probs = F.softmax(logits, dim=-1)
            
            # 这里我们用 Top-k 采样，让句子稍微通顺点
            v, i = torch.topk(logits, 10)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            print("next token: \n ",next_token)
            print("next token item: \n ",[next_token.item()])
            
            x = torch.cat((x, next_token), dim=1)
            
            # 实时打印（如果 flush 可用）
            print(decode([next_token.item()]), end='', flush=True)

    print("\n\n生成结束！")