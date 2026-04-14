import torch
import torch.nn as nn
import torch.nn.functional as F

# 🔥 PATH TO YOUR MODEL
MODEL_PATH = r"C:\Users\kaden\Downloads\mini_ai_model (1).pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

state_dict = checkpoint["model_state_dict"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
config = checkpoint.get("config", {})

vocab_size = len(stoi)

# -----------------------------
# MODEL DEFINITION (MATCH TRAINING)
# -----------------------------
N_EMBD = config.get("n_embd", 512)
N_HEAD = config.get("n_head", 16)
N_LAYER = config.get("n_layer", 8)
BLOCK_SIZE = config.get("block_size", 128)
DROPOUT = config.get("dropout", 0.2)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = torch.tril(torch.ones(T, T, device=x.device)) * wei
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx

# -----------------------------
# LOAD MODEL
# -----------------------------
model = MiniTransformer().to(DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

# -----------------------------
# TEXT GENERATION
# -----------------------------
def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=DEVICE).unsqueeze(0)

def decode(t):
    return ''.join([itos[i] for i in t])

prompt = input("Enter prompt: ")

context = encode(prompt)
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)

print("\n--- OUTPUT ---\n")
print(decode(generated[0].tolist()))
