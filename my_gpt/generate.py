import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import os

def generate():
    # -----------------------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    
    load_pretrained = True # set to False to load from ckpt.pt
    # -----------------------------------------------------------------------------

    if load_pretrained:
        model = GPT.from_pretrained('gpt2')
    else:
        # Load from local checkpoint
        # Note: You need to know the config used for training!
        # This is a bit tricky if we don't save config. 
        # For simplicity, let's assume a default small config matching train.py
        # in a real scenario, save config.json with the checkpoint.
        print("Loading from local checkpoint...")
        config = GPTConfig(
            block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.0 # dropout usually 0 for inference
        )
        # We need to know vocab_size too. 
        # Ideally, we load this from metadata.
        # For now, let's assume we are using the same input.txt and tokenizer logic,
        # OR we just fail if we can't reconstruct it.
        # To make this robust, let's verify if input.txt exists to rebuild vocab
        try:
            with open('input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            chars = sorted(list(set(text)))
            config.vocab_size = len(chars)
        except Exception:
            print("Could not infer vocab size from input.txt, assuming default 50257 (GPT-2)")
            config.vocab_size = 50257

        model = GPT(config)
        if os.path.exists('ckpt.pt'):
            model.load_state_dict(torch.load('ckpt.pt', map_location=device))
        else:
            print("ckpt.pt not found! Initializing random model.")
    
    model.to(device)
    model.eval()

    # Tokenizer set up
    # If loading pretrained GPT-2, use tiktoken if available
    if load_pretrained:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
        except ImportError:
            print("tiktoken not installed. Taking a shortcut... (not recommended for pretrained)")
            # Fallback (very bad for pretrained, but just so it runs)
            encode = lambda s: [ord(c) for c in s]
            decode = lambda l: ''.join([chr(i) for i in l])
    else:
        # Custom tokenizer
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
    
    # Prompt
    prompt = "Today I want to have something for lunch, and here is my choice: "
    print(f"\nPrompt: {prompt}")
    
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    print("Generating...")
    with torch.no_grad():
        for k in range(50):
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Top-k sampling
            v, i = torch.topk(logits, 10)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
            
            try:
                print(decode([next_token.item()]), end='', flush=True)
            except Exception:
                pass # ignore decode errors

    print("\n\nDone!")

if __name__ == '__main__':
    generate()
