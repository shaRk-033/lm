import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# with open('input.txt', 'r') as f:
#     data = f.read()

# vocab = sorted(list(set(data)))
# encode = {char: i for i, char in enumerate(vocab)}
# decode = {i: char for i, char in enumerate(vocab)}
# enc_data = [encode[char] for char in data]
ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
tokenizer = tiktoken.get_encoding('gpt2')
data = "\n".join(item['text'] for item in ds)
enc_data = tokenizer.encode(data)
da = torch.tensor(enc_data).to(device)

def train_val_split(data, split):
    d = data[:int(len(data) * 0.8)]
    if split == 'train': return d
    elif split == 'val': return data[int(len(data) * 0.8):int(len(data) * 0.9)]
    elif split == 'test': return data[int(len(data) * 0.9):]

def gen_batch(data, batch_size, context):
    i = torch.randint(0, len(data) - context, (batch_size,), device=device)
    x = torch.stack([data[j:j+context] for j in i])
    y = torch.stack([data[j+1:j+context+1] for j in i])
    return x, y

class Config:
    vocab_size = 50304
    d_model = 64
    batch_size = 32
    context = 16
    n_heads = 8
    n_groups = 4
    n_layers = 2

config = Config()

# arXiv:2104.09864v5
class RMSNorm(nn.Module):
    def __init__(self) -> None:
        super(RMSNorm, self).__init__()
        self.alpha = 0.99
        self.eps = 1e-8
    def forward(self, x):
        B, T, C = x.shape
        rms = (torch.mean(x**2, dim = -1, keepdim = True))**0.5
        x = x / (rms + self.eps)
        return x

# arXiv:2002.05202v1
class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model) -> None:
        super(FFN_SwiGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_model*4, bias = False)
        self.w2 = nn.Linear(d_model * 4, d_model, bias = False)
        self.beta = nn.Parameter(torch.ones(1))
        self.v = nn.Linear(d_model, d_model * 4, bias = False)

    def forward(self, x):
        var1 = self.w1(x)
        var2 = self.v(x)
        swish = var1 * torch.sigmoid(var1 * self.beta)
        gate_out = swish * var2
        x = self.w2(gate_out)
        return x

# group query attention and ROPE
class SelfAttn(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n = config.n_heads
        self.ng = config.n_groups
        self.h = config.d_model // self.n
        self.gh = (config.d_model // self.n) * config.n_groups
        self.q = nn.Linear(config.d_model, config.d_model, bias = False)
        self.k = nn.Linear(config.d_model, self.gh, bias = False)
        self.v = nn.Linear(config.d_model, self.gh, bias = False)

    def get_rotary_matrix(self, config):
        ro_matrix = torch.zeros((config.context, config.d_model, config.d_model), device=device)
        r_k = torch.zeros((config.context, self.gh, self.gh), device=device)
        for pos in range(config.context):
            for i in range(0, config.d_model//2):
                theta = 10000 ** -(2 *(i - 1))/config.d_model
                ro_matrix[pos, 2*i, 2*i] = np.cos(pos * theta)
                ro_matrix[pos, 2*i, 2*i+1] = -np.sin(pos * theta)
                ro_matrix[pos, 2*i+1, 2*i] = np.sin(pos * theta)
                ro_matrix[pos, 2*i+1, 2*i+1] = np.cos(pos * theta)
            for i in range(0, self.gh//2):
                theta = 10000 ** -(2 *(i - 1))/(self.gh)
                r_k[pos, 2*i, 2*i] = np.cos(pos * theta)
                r_k[pos, 2*i, 2*i+1] = -np.sin(pos * theta)
                r_k[pos, 2*i+1, 2*i] = np.sin(pos * theta)
                r_k[pos, 2*i+1, 2*i+1] = np.cos(pos * theta)
        return ro_matrix, r_k

    def forward(self, x):
        b,t,c = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        n = self.n
        h = self.h
        g = self.gh
        ng = self.ng

        r, rk = self.get_rotary_matrix(config)
        q = torch.bmm(q.transpose(0, 1), r).transpose(0, 1)
        k = torch.bmm(k.transpose(0, 1), rk).transpose(0, 1)
        
        q = q.transpose(1,2).view(b,n,h,t)
        k = k.transpose(1,2).view(b,ng,h,t)
        v = v.transpose(1,2).view(b,ng,h,t)

        q = q.view(b, ng, n//ng, h, t)
        k = k.view(b, ng, 1, h, t)
        v = v.view(b, ng, 1, h, t)

        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)

        att = (q @ k.transpose(-1,-2)) / (h ** 0.5)
        mask = torch.tril(torch.ones(t, t)).view(1, 1, t, t).to(device)
        att = att.masked_fill(mask == 0, float('-inf'))
        self_attn = F.softmax(att, dim = -1)

        attn = (self_attn @ v.transpose(-1,-2)).transpose(1, 2).contiguous().view(b, t, c)
        return attn

class DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderLayer, self).__init__()
        self.attn = SelfAttn(config)
        self.rmsn1 = RMSNorm()
        self.ffn = FFN_SwiGLU(config.d_model)
        self.rmsn2 = RMSNorm()
    def forward(self, x):
        x1 = self.attn(self.rmsn1(x)) + x
        x2 = self.ffn(self.rmsn2(x1)) + x1
        return x2

class Shark(nn.Module):
    def __init__(self, config):
        super(Shark, self).__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.final = nn.Linear(config.d_model, config.vocab_size)
        print("Number of params: ", sum(p.numel() for p in self.parameters()))
    def forward(self, x, targets = None):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, config.vocab_size), targets.view(-1))
            return x, loss
        else:
            return x
    # def generate(self, start_prompt, max_length=100):
    #     self.eval()
    #     input_ids = [encode[char] for char in start_prompt] 
    #     input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) 
        
    #     generated = input_ids[:] 
    #     for _ in range(max_length):
    #         x = input_tensor[:, -config.context:] 
    #         logits = self(x)[:, -1, :] 
    #         probs = F.softmax(logits, dim=-1) 
            
    #         next_token = torch.multinomial(probs, num_samples=1).item() 
    #         generated.append(next_token) 
            
    #         input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
            
    #     generated_text = ''.join([decode[token] for token in generated])
    #     return generated_text

model = Shark(config).to(device)
opt = optim.Adam(model.parameters(), lr=0.001)

@torch.no_grad()
def validate(model, data, config):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        split_data = train_val_split(data, split)
        for _ in range(10):
            xb, yb = gen_batch(split_data, config.batch_size, config.context)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

def train(data, model, opt, config, epochs=3000):
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        x, y = gen_batch(data, config.batch_size, config.context)
        y_pred, loss = model(x, y)
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}')
            x = validate(model, data, config)
            losses += [x]
            print(f'Train loss: {x["train"]}, Val loss: {x["val"]}')
    return pd.DataFrame(losses).plot()

try:
    train(da, model, opt, config)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("CUDA out of memory. Try reducing batch size or model size.")
    raise e
