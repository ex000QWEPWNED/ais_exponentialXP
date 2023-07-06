!wget https://raw.githubusercontent.com/ex000QWEPWNED/ais_exponentialXP/main/input.txt
!pip install transformers
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer
import random
import time

# Initialize the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# hyperparameters
batch_size = 32   # Depending on your GPU memory size. Larger batches offer more stable gradients, but smaller batches use less memory.
block_size = 128  # Depending on the context length of your tasks. Larger block sizes may improve performance, but at the cost of increased memory usage.
max_iters = 250  # Depending on how large and complex your data is.
eval_interval = 10  # It's usually good to check the performance more frequently at the beginning and then reduce the frequency.
learning_rate = 1e-3  # You might need to adjust this based on your model's performance. Smaller learning rates usually lead to more stable training but slower convergence.
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # This should remain as it is.
eval_iters = 10  # This is to reduce the time taken in evaluation phase. Depending on your needs, you might want to adjust this.
n_embd = 64 # This should be fine for a smaller model. Consider increasing it if you want the model to learn more complex representations.
n_head = 4 # This should be fine for a smaller model. More heads would allow the model to focus on different aspects of the input.
n_layer = 4  # This should be fine for a smaller model. More layers would allow the model to model more complex interactions.
dropout = 0.2 # Typically ranges from 0.1 to 0.5. Higher values regularize more but can limit model power.

# ------------

print(device)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get vocabulary size from tokenizer
vocab_size = tokenizer.vocab_size

# create a mapping from tokens to integers using the tokenizer
encode = lambda s: tokenizer.encode(s) # encoder: take a string, output a list of integers
decode = lambda l: tokenizer.decode(l) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# Set a seed
random.seed(42)

def get_sequences(data, seq_len):
    """ Split data into sequences of a specific length """
    return [data[i:i+seq_len] for i in range(0, len(data), seq_len)]

# Get sequences
seq_len = block_size
sequences = get_sequences(data, seq_len)

# Shuffle sequences
random.shuffle(sequences)

# Concatenate sequences to form shuffled data
shuffled_data = torch.cat(sequences)

# Update the train and test splits
n = int(0.9*len(shuffled_data))
train_data = shuffled_data[:n]
val_data = shuffled_data[n:]


# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
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

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1):
    # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
        # get the predictions
            logits, loss = self(idx_cond)
        # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Start the overall timer
start_time = time.time()

for iter in range(max_iters):

    # check if user wants to continue training every 1000 steps
    if iter % 20000 == 0 and iter > 0:
        inp = input('Do you want to continue training? (yes/no): ')
        if inp.lower() == 'no':
            # save model state and break loop
            torch.save(model.state_dict(), r"my_model.pt")
            print("Model saved and training stopped.")
            break

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        # Calculate the elapsed time for these iterations
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Reset the start time for the next set of iterations
        start_time = time.time()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


input_text = "He"
context = torch.tensor(encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(context, max_new_tokens=5)
generated_text = tokenizer.decode(generated[0].tolist())
print("Generated text:")
print(generated_text)

