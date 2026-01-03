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
        self.MLP = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd,bias = config.bias)

    def forward(self,x):
        # residual of attention
        x = x + self.attn(self.ln_1(x))
        # residual of MLP
        x = x + self.MLP(self.ln_2(x))

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
    # 0. 检查设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps' # <--- 这里激活 Apple Silicon 的 GPU    print(f"正在使用设备: {device}")

    # 1. 修改配置以适应我们的微型数据
    # 注意：这里的 vocab_size 必须等于我们上面算出来的 len(chars)
    config = GPTConfig(
        vocab_size = vocab_size, # 比如 65
        block_size = 256,        # 上下文长度
        n_layer = 6,             # 稍微小一点的模型
        n_head = 6,
        n_embd = 384,
        dropout = 0.2
    )
    
    # 2. 初始化模型
    model = GPT(config)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"device type:{device}")

    # 3. 创建优化器 (AdamW)
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. 训练循环
    max_iters = 5000
    eval_interval = 5
    batch_size = 64
    block_size = config.block_size
   
    print("开始训练...")
    last_time = time.time()
    

    for iter in range(max_iters):

        # 每隔一段时间评估一下 loss，看看有没有进步
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, 1, block_size, batch_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time cost: {time.time()-last_time}")
            last_time = time.time()

        # --- 核心训练步骤 ---
        # A. 拿数据
        xb, yb = get_batch('train', block_size, batch_size, device)

        # B. 前向传播
        logits = model(xb)
        
        # C. 计算 Loss
        # PyTorch 的 CrossEntropy 需要 (N, C) 形状，所以要 reshape
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

        # D. 反向传播 (三板斧)
        optimizer.zero_grad(set_to_none=True) # 清空梯度
        loss.backward()                       # 计算梯度
        optimizer.step()                      # 更新参数

    print(f"训练结束！耗时: {time.time() - start_time:.2f}s")

    # 5. 生成文本测试 (Inference)
    print("\n--- 生成文本 ---")
    # 这里的 context 是一个全 0 的序列 (通常代表换行符或开始)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # 这是一个简单的生成循环
    model.eval()
    generated_chars = []
    with torch.no_grad():
        for _ in range(500): # 生成 500 个字
            # 1. 获取最后 block_size 个 token
            idx_cond = context[:, -block_size:]
            # 2. 模型预测
            logits = model(idx_cond)
            # 3. 只要最后一个时间步的 logits
            logits = logits[:, -1, :] 
            # 4. 转换成概率
            probs = F.softmax(logits, dim=-1)
            # 5. 采样 (Sample)
            idx_next = torch.multinomial(probs, num_samples=1)
            # 6. 拼接到 context
            context = torch.cat((context, idx_next), dim=1)
            # 7. 记录
            generated_chars.append(idx_next.item())

    print(decode(generated_chars))