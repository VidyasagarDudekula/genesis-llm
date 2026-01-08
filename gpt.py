import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
BLOCK_SIZE = 512

with open('input.txt', 'r') as file:
    shake_sp_data = file.read()
with open('grimms.txt', 'r') as file:
    grimms_data = file.read()
with open('sherlock.txt', 'r') as file:
    sherlocks_data = file.read()
with open('dracula.txt', 'r') as file:
    dracula_data = file.read()

token_encoder = tiktoken.encoding_for_model("gpt-4o")

total_text_data = f"""
Poems:-

{shake_sp_data}


END of Poems

{" "*BLOCK_SIZE}

New Content:-

stories:-

{grimms_data}

END of Stories

{" "*BLOCK_SIZE}

New Content:-

Sherlocks stories:-

{sherlocks_data}

End of Sherlocks stories.

{" "*BLOCK_SIZE}

New Content:-

Dracula Stories:-

{dracula_data}

"""

# global satic values
data_length = len(total_text_data)
train_split = int(data_length * 0.97)
train_total_tokens = token_encoder.encode(total_text_data[:train_split])
train_total_tokens_tensor = torch.tensor(train_total_tokens)
val_total_tokens = token_encoder.encode(total_text_data[train_split:])
val_total_tokens_tensor = torch.tensor(val_total_tokens)
vocab_tokens = sorted(list(set(token_encoder.encode(total_text_data))))
VOCAB_SIZE = token_encoder.n_vocab
EMBED_SIZE = 256
BATCH_SIZE = 32
NUM_HEADS = 4
assert EMBED_SIZE % NUM_HEADS == 0
HEAD_SIZE = EMBED_SIZE//NUM_HEADS
DECODER_LAYERS = 2
DROPOUT = 0.2
ITER = 2000


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(EMBED_SIZE, 4*EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE*4, EMBED_SIZE),
            nn.Dropout(DROPOUT)
        )
    
    def forward(self, x):
        out = self.layer(x)
        return out


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(EMBED_SIZE, HEAD_SIZE, bias=False)
        self.query = nn.Linear(EMBED_SIZE, HEAD_SIZE, bias=False)
        self.value = nn.Linear(EMBED_SIZE, HEAD_SIZE, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.float, device=device)))
    
    def forward(self, input_embeddings):
        B, T, C = input_embeddings.shape
        q = self.query(input_embeddings)
        k = self.key(input_embeddings)
        similarity = q @ k.transpose(-2, -1)/(HEAD_SIZE**0.5)
        wei = similarity.masked_fill(self.tril[:T, :T]==0.0, value=float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(input_embeddings)
        out = wei @ v
        return out


class MultiHeadAttentionCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(NUM_HEADS)])
        self.WH = nn.Linear(EMBED_SIZE, EMBED_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, inputs):
        out = torch.cat([head(inputs) for head in self.heads], dim=-1)
        out = self.WH(out)
        out = self.dropout(out)
        return out
    
class SelfAttentionDecoder(nn.Module):
    def __init__(self):
        self.key = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.query = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.value = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self, input_embddings):
        B, T, C = input_embddings.shape
        q = self.query(input_embddings).reshape(B, T, NUM_HEADS, HEAD_SIZE).transpose(1, 2) # B, NUM_HEADS, T, HEAD_SIZE
        k = self.key(input_embddings).reshape(B, T, NUM_HEADS, HEAD_SIZE).transpose(1, 2) # B, NUM_HEADS, T, HEAD_SIZE
        v = self.value(input_embddings).reshape(B, T, NUM_HEADS, HEAD_SIZE).transpose(1, 2) # B, NUM_HEADS, T, HEAD_SIZE
        qk = q @ k.transpose(-2, -1)
        qk *= HEAD_SIZE ** -0.5
        
    

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mhc_attention = MultiHeadAttentionCausal()
        self.ff_layer = FeedForward()
        self.ln1 = nn.LayerNorm(EMBED_SIZE)
        self.ln2 = nn.LayerNorm(EMBED_SIZE)
    
    def forward(self, input_embeddings):
        input_embeddings = input_embeddings + self.mhc_attention(self.ln1(input_embeddings))
        input_embeddings = input_embeddings + self.ff_layer(self.ln2(input_embeddings))
        return input_embeddings


class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, EMBED_SIZE, device=device)
        self.pos_embedding_table = nn.Embedding(BLOCK_SIZE, EMBED_SIZE, device=device)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock() for _ in range(DECODER_LAYERS)])
        self.ln_final = nn.LayerNorm(EMBED_SIZE)
        self.layer = nn.Linear(EMBED_SIZE, VOCAB_SIZE)
    
    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        input_embeddings = self.embeddings(inputs) # B, T, C
        pos_embed = torch.arange(0, T, device=device)
        pos_embed = self.pos_embedding_table(pos_embed)
        input_embeddings += pos_embed
        input_embeddings = self.decoder_blocks(input_embeddings)
        input_embeddings = self.ln_final(input_embeddings)
        logits = self.layer(input_embeddings)
        if targets is None:
            return logits, None
        logits_view = logits.view(B*T, VOCAB_SIZE)
        targets_view = targets.view(B*T,)
        loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss
    
    def generate(self, inputs, max_token_length=100):
        window = inputs[:]
        for _ in range(max_token_length):
            if window.shape[1]>BLOCK_SIZE:
                window = window[:, window.shape[1]-BLOCK_SIZE:, ]
            logits, _ = self(window)
            last_channel = logits[:, -1, :]
            last_channel = F.softmax(last_channel, dim=1)
            next_tokens = torch.multinomial(last_channel, num_samples=1)
            window = torch.cat((window, next_tokens), dim=-1)
            inputs = torch.cat((inputs, next_tokens), dim=-1)
        return inputs
        
    

def get_batch(split="train"):
    target_token = train_total_tokens_tensor if split == "train" else val_total_tokens_tensor
    ix = torch.randint(0, target_token.size()[0]-BLOCK_SIZE, (BATCH_SIZE,))
    xb = torch.stack([target_token[i: i+BLOCK_SIZE] for i in ix]).to(device)
    yb = torch.stack([target_token[i+1: i+BLOCK_SIZE+1] for i in ix]).to(device)
    return xb, yb

# test_head = torch.randn((BATCH_SIZE, BLOCK_SIZE, EMBED_SIZE))
# head = Head()
# out = head(test_head)
# print(out.shape)

if __name__ == "__main__":
    model = LLM()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    stepi = []
    lossi = []
    for step in range(ITER):
        xb, yb = get_batch()
        model.zero_grad(set_to_none=True)
        logits, loss = model(xb, yb)
        if step%50 ==0:
            print(f"Step {step}: Loss {loss.item():.4f}")
            stepi.append(step)
            lossi.append(loss.item())
        loss.backward()
        optimizer.step()
    torch.save(model, "model.pth")
    print(loss.item())
    
    plt.plot(stepi, lossi)
    plt.title("Training Loss")
    plt.xlabel("Step (x50)")
    plt.ylabel("Loss")
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")
        
    torch.manual_seed(37)
    test_data = torch.tensor([token_encoder.encode("Once upon a time:- ")], device=device)
    gen_data_tensors = model.generate(test_data)
    gen_data = [token_encoder.decode(x.tolist()) for x in gen_data_tensors]
    print(gen_data[0])
