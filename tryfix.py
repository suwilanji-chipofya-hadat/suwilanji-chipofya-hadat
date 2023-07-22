import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()

		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads

		self.W_q = nn.Linear(d_model, d_model)
		self.W_k = nn.Linear(d_model, d_model)
		self.W_v = nn.Linear(d_model, d_model)
		self.W_o = nn.Linear(d_model, d_model)
	def scaled_dot_product_attention(self, Q, K, V, mask=None):
		attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
		if mask is not None:
			attention_scores = attention_scores.masked_fill(mask==0, -1e9)
		attention_probs = torch.softmax(attention_scores, dim=1)
		output = torch.matmul(attention_probs, V)

		return output

	def split_heads(self,x):
		batch_size, seq_len, d_model = x.size()
		return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
	def combine_heads(self, x):
		batch_size, _, seq_len, d_k = x.size()
		return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
	def forward(self, Q, K, V, mask=None):
		Q = self.split_heads(self.W_q(Q))
		K = self.split_heads(self.W_k(K))
		V = self.split_heads(self.W_v(V))

		attention_output = self.scaled_dot_product_attention(Q, K,V, mask)
		output = self.W_o(self.combine_heads(attention_output))
		return output

class PositionWiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff):
		super(PositionWiseFeedForward, self).__init__()
		self.fc1 = nn.Linear(d_model, d_ff)
		self.fc2 = nn.Linear(d_ff, d_model)
		self.relu = nn.ReLU()

	def forward(self, x):
		return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_seq_len):

		super(PositionalEncoding, self).__init__()

		pe = torch.zeros(max_seq_len, d_model)
		position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float()*-(math.log(10000.0) / d_model))

		pe[:, 0::2] = torch.sin(position*div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		self.register_buffer('pe', pe.unsqueeze(0))
	def forward(self, x):
		return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):

	def __init__(self, d_model, num_heads,d_ff, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(d_model, num_heads)
		self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, mask=None):
		attn_output = self.self_attn(x,x,x,mask)
		x = self.norm1(x+self.dropout(attn_output))
		ff_output = self.feed_forward(x)
		x = self.norm2(x + self.dropout(ff_output))

		return x

class DecoderLayer(nn.Module):

	def __init__(self, d_model, num_heads,d_ff,dropout):
		super(DecoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(d_model, num_heads)
		self.cross_attn = MultiHeadAttention(d_model, num_heads)
		self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
		attn_output = self.self_attn(x,x,x,tgt_mask)
		x = self.norm1(x + self.dropout(attn_output))
		attn_output = self.cross_attn(x,enc_output,enc_output,src_mask)
		x = self.norm2(x + self.dropout(attn_output))
		ff_output = self.feed_forward(x)
		x = self.norm3(x + self.dropout(ff_output))

		return x

class Transformer(nn.Module):

	def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout):
		super(Transformer, self).__init__()
		self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
		self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
		self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

		self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
		self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

		self.fc = nn.Linear(d_model, tgt_vocab_size)
		self.dropout = nn.Dropout(dropout)

	def generate_mask(self, src, tgt):

		src_mask = (src!=0).unsqueeze(1).unsqueeze(2)
		tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

		seq_len = tgt.size(1)
		nopeak_mask = (1-torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
		tgt_mask = tgt_mask & nopeak_mask
		return src_mask, tgt_mask
	def forward(self, src, tgt):
		src_mask, tgt_mask = self.generate_mask(src, tgt)

		src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
		tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

		enc_output = src_embedded

		for enc_layer in self.encoder_layers:
			enc_output = enc_layer(enc_output, src_mask)

		dec_output = tgt_embedded
		for dec_layer in self.decoder_layers:
			dec_output = dec_layer(dec_output,enc_output, src_mask,tgt_mask)

		output = self.fc(dec_output)
		return output

# Testing

# Generate random sample data
text = None
with open("english_text.txt", "r") as f: text=f.read()
src_data = CharacterEncoder(text)
src_vocab_size = len(src_data.chars)
src_data = src_data.encode()

text1 = None
with open("english_text.txt", "r") as f: text1=f.read()
tgt_data = CharacterEncoder(text)
tgt_vocab_size = len(tgt_data.chars)
tgt_data = tgt_data.encode()
#src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
#tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

import numpy as np
src_num_rows = len(src_data) // 100
src_2d = np.array(src_data[:src_num_rows * 100]).reshape(src_num_rows, 100)
remaining_items = src_data[src_num_rows *100:]
if remaining_items:
  padding = [0] * (100-len(remaining_items))
  padded_row = remaining_items + padding
  src_2d = np.vstack([src_2d, padded_row])
tgt_num_rows = len(tgt_data) // 100
tgt_2d = np.array(tgt_data[:tgt_num_rows * 100]).reshape(tgt_num_rows, 100)
remaining_items = tgt_data[tgt_num_rows *100:]
if remaining_items:
  padding = [0] * (100-len(remaining_items))
  padded_row = remaining_items + padding
  tgt_2d = np.vstack([tgt_2d, padded_row])

src_2d = torch.tensor(src_2d)
tgt_2d = torch.tensor(tgt_2d)
data_x = src_2d
data_y = tgt_2d
'''n = int(len(data) * 0.9)
train_datax = data_x[:n]
train_datay = data_y[:n]
val_datax = data_x[n:]
val_datay = data_y[n:]'''
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    #data_x = train_datax if split == 'train' else val_datax
    global data_x
    #data_y = train_datay if split == 'train' else val_datay
    global data_y
    ix = torch.randint(len(data) - block_size - 1, (batch_size,)) #randomly select starting indices
    x= torch.stack([data_x[i: i + block_size] for i in ix]) #input sequence
    y = torch.stack([data_y[i + 1: i + block_size + 1] for i in ix]) #target sequence
    x,y  = x.to(device), y.to(device)
    return x,y

print(src_2d)
print(src_2d.shape)
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
print(f'number of parameters: {sum(p.numel() for p in transformer.parameters())}')
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_2d, tgt_2d)
    print(f"Output: {output.contiguous().view(-1, tgt_vocab_size)}\nTarget:{tgt_data[:, 1:].contiguous().view(-1)}")
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
# Import statements and classes...

# Define the CharacterEncoder class to handle text-to-index encoding
class CharacterEncoder:
    def __init__(self, text):
        self.chars = list(set(text))
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(self.chars)}

    def encode(self):
        return [self.char_to_idx[ch] for ch in text]

# Rest of the code...
# (Note that I'm skipping the testing part since the issues are in the training part)

# Define the get_batch function to generate input and target batches
def get_batch(data_x, data_y, batch_size, block_size):
    ix = torch.randint(len(data_x) - block_size - 1, (batch_size,))
    x = torch.stack([data_x[i: i + block_size] for i in ix])
    y = torch.stack([data_y[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Generate random sample data
text = None
with open("english_text.txt", "r") as f:
    text = f.read()
src_data = CharacterEncoder(text)
src_vocab_size = len(src_data.chars)
src_data = src_data.encode()

text1 = None
with open("english_text.txt", "r") as f:
    text1 = f.read()
tgt_data = CharacterEncoder(text1)
tgt_vocab_size = len(tgt_data.chars)
tgt_data = tgt_data.encode()

# Convert data to tensors and reshape
src_2d = torch.tensor(src_data)
tgt_2d = torch.tensor(tgt_data)

# Assuming the max_seq_length is 100 as defined in the Transformer class
# Make sure your data has a length that is a multiple of max_seq_length

# Data reshaping to 3D tensors (batch_size, seq_length, input_dim)
src_3d = src_2d.view(-1, max_seq_length, 1)
tgt_3d = tgt_2d.view(-1, max_seq_length, 1)

# The rest of the code...

# Initialize the Transformer model and optimizer
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Training loop
transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_3d, tgt_3d)
    output_flattened = output.contiguous().view(-1, tgt_vocab_size)
    target_flattened = tgt_3d[:, 1:].contiguous().view(-1)
    loss = criterion(output_flattened, target_flattened)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

