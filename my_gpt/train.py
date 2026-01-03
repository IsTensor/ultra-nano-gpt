import torch
from model import GPT, GPTConfig
import time

def train():
    # -----------------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------------
    batch_size = 16 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 10
    eval_interval = 2
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    eval_iters = 5
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    # -----------------------------------------------------------------------------

    print(f"Using device: {device}")
    torch.manual_seed(1337)

    # Load data
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("input.txt not found. Using a dummy string for demonstration.")
        text = "Hello world! This is a dummy dataset for testing the training loop." * 1000

    # Tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Train/Val split
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Data Loader
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    # Estimate loss
    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Model initialization
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )
    model = GPT(config)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Training model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Training Loop
    start_time = time.time()
    for iter in range(max_iters):
        # Evaluation
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Forward pass
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Save model
    torch.save(model.state_dict(), 'ckpt.pt')
    print("Model saved to ckpt.pt")

if __name__ == '__main__':
    train()
