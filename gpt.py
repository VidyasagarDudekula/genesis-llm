import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader_files import total_text_data, complete_text_data, validation_data

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
BLOCK_SIZE = 768

token_encoder = tiktoken.encoding_for_model("gpt-3.5")
total_set = set(token_encoder.encode(complete_text_data)) | set(token_encoder.encode(total_text_data)) | set(token_encoder.encode(validation_data))
vocab_tokens = sorted(list(total_set))
ttoi = {t:i for i,t in enumerate(vocab_tokens)}
itot = {i:t for i,t in enumerate(vocab_tokens)}

def token_to_index(token):
    return ttoi[token]

def index_to_token(index):
    return itot[index]

def encode(content):
    token_ids = token_encoder.encode(content)
    return list(map(token_to_index, token_ids))

def decode(indices):
    token_ids = list(map(index_to_token, indices))
    return token_encoder.decode(token_ids)

# global satic values
data_length = len(total_text_data)
train_total_tokens = encode(total_text_data)
train_total_tokens_tensor = torch.tensor(train_total_tokens)
val_total_tokens = encode(validation_data)
val_total_tokens_tensor = torch.tensor(val_total_tokens)
VOCAB_SIZE = len(vocab_tokens)
print(VOCAB_SIZE, train_total_tokens_tensor.max(), val_total_tokens_tensor.max())
EMBED_SIZE = 384
BATCH_SIZE = 32
NUM_HEADS = 6
assert EMBED_SIZE % NUM_HEADS == 0
HEAD_SIZE = EMBED_SIZE//NUM_HEADS
DECODER_LAYERS = 8
DROPOUT = 0.2
ITER = 5000


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


    
class SelfAttentionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.query = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.value = nn.Linear(EMBED_SIZE, EMBED_SIZE, bias=False)
        self.linear_layer = nn.Linear(EMBED_SIZE, EMBED_SIZE)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, device=device, dtype=torch.float)))
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, input_embddings):
        B, T, C = input_embddings.shape
        q = self.query(input_embddings).reshape(B, T, NUM_HEADS, HEAD_SIZE).transpose(1, 2) # B, NUM_HEADS, T, HEAD_SIZE
        k = self.key(input_embddings).reshape(B, T, NUM_HEADS, HEAD_SIZE).transpose(1, 2) # B, NUM_HEADS, T, HEAD_SIZE
        v = self.value(input_embddings).reshape(B, T, NUM_HEADS, HEAD_SIZE).transpose(1, 2) # B, NUM_HEADS, T, HEAD_SIZE
        qk = q @ k.transpose(-2, -1) # B, NUM_HEADS, T, T
        qk *= HEAD_SIZE ** -0.5
        attns = qk.masked_fill(self.tril[:T, :T]==0.0, value=float('-inf'))
        attns = F.softmax(attns, dim=-1)
        attns = attns @ v # B, NUM_HEADS, T, HEAD_SIZE
        attns = attns.transpose(1, 2).contiguous().view(B, T, C)
        out = self.linear_layer(attns)
        out = self.dropout(out)
        return out
        
    

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention_decoder = SelfAttentionDecoder()
        self.ff_layer = FeedForward()
        self.ln1 = nn.LayerNorm(EMBED_SIZE)
        self.ln2 = nn.LayerNorm(EMBED_SIZE)
    
    def forward(self, input_embeddings):
        input_embeddings = input_embeddings + self.self_attention_decoder(self.ln1(input_embeddings))
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
        self.embeddings.weight = self.layer.weight
    
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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200) # Average over 200 batches
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    model = LLM()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    stepi = []
    lossi = []
    valii = []
    for step in range(ITER):
        xb, yb = get_batch()
        model.zero_grad(set_to_none=True)
        logits, loss = model(xb, yb)
        if step%50 ==0:
            loss_check = estimate_loss()
            print(f"Step {step}: Train Loss {loss_check['train']:.4f}, Val Loss {loss_check['val']:.4f}")
            stepi.append(step)
            lossi.append(loss_check['train'])
            valii.append(loss_check['val'])
        loss.backward()
        optimizer.step()
    torch.save(model, "model.pth")
    print(loss.item())
    
    plt.figure(figsize=(10, 6))
    plt.plot(stepi, lossi, label='Train Loss')
    plt.plot(stepi, valii, label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend() 
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")
        
    torch.manual_seed(37)
    test_data = torch.tensor([encode("Once upon a time:- ")], device=device)
    gen_data_tensors = model.generate(test_data)
    gen_data = [decode(x.tolist()) for x in gen_data_tensors]
    print(gen_data[0])
