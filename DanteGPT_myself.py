import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper function to move things easily to the device (we are not using it right now)

def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs).to(device)

# Importing data from a txt file

with open("commedia_cleaned.txt", "r", encoding="utf-8") as f:
    corpus = f.read()
    text = corpus[52:]

# Creating the dataset and functions to handle it

chars = sorted(list(set(text)))
vocab_size = len(chars)
batch_size = 32
block_size = 8
eval_intervals = 500
eval_iters = 200
max_iters = 5000
n_embd = 64
dropout = 0.2
num_heads = 4
num_layers = 6
learning_rate = 1e-3

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[n] for n in l])
data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_set = data[:n] 
val_set = data[n:]

def get_batch(split):
    data = train_set if split=="train" else val_set
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x.to(device), y.to(device)
    return x, y

# Creating a function to evaluate loss more accurately by averaging losses on a set of eval_iters; we will run this every eval_intervals iterations for both the train set and the eval set

@torch.no_grad
def eval_loss():
    out = {}
    model.eval()
    splits = ["train", "eval"]
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Xb, Yb = get_batch(split)    
            logits, loss = model(Xb, Yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Let's define the classes that create the model of our deep neural network

class Head(nn.Module):
    """ one self-attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(head_size)
        self.ffw = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        emb_tok = self.token_embedding_table(idx)
        emb_pos = self.positional_embedding_table(torch.arange(T, device=device))
        x = emb_tok + emb_pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1,replacement=False)
            idx = torch.cat((idx,idx_next),dim=-1)
        return idx


def train_model(model, optimizer):

    for iter in range(max_iters):

        if iter % eval_intervals == 0 or iter == max_iters - 1:
            losses = eval_loss()
            print(f"Step {iter}: train loss {losses['train']:.4f} and eval loss {losses['eval']:.4f}")

        Xb, Yb = get_batch("train")
        logits, loss = model(Xb, Yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def decode_new_tokens(model, context):

    output = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
    print(output)

if __name__ == '__main__':
    model = BigramLanguageModel()
    model = model.to(device)
    
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate new tokens')
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to save or load the model')
    args = parser.parse_args()
    
    if args.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train_model(model, optimizer)
        torch.save(model.state_dict(), args.model_path)
        print(f"Trained model saved to {args.model_path}")
    
    if args.generate:
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path))
            print(f"Loaded trained model from {args.model_path}")
        else:
            print(f"No trained model found at {args.model_path}. Using untrained model.")
        
        starter_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        decode_new_tokens(model, starter_token)