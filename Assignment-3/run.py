import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import math
import csv

# Parameters
SEED = 42
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 30
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Set seed for reproducibility
torch.manual_seed(SEED)

# Load dataset
def load_data():
    ptb = load_dataset("ptb-text-only/ptb_text_only")
    return ptb["train"]["sentence"], ptb["validation"]["sentence"], ptb["test"]["sentence"]

# Preprocess sentences
def preprocess_sentences(sentences):
    return [sentence.lower() for sentence in sentences]

# Vocabulary Class
class Vocabulary:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = SPECIAL_TOKENS

    def build_vocab(self, sentences):
        word_counter = Counter(word for sentence in sentences for word in sentence.split())
        most_common = self.special_tokens + [word for word, _ in word_counter.most_common(self.vocab_size - len(self.special_tokens))]
        self.word2idx = {word: idx for idx, word in enumerate(most_common)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, sentence):
        tokens = ["<bos>"] + sentence.split() + ["<eos>"]
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

    def decode(self, indices):
        return " ".join(self.idx2word[idx] for idx in indices)

# PTB Dataset Class
class PTBDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_sentences, max_seq_len):
        self.data = tokenized_sentences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_seq = tokens[:-1][:self.max_seq_len]
        target_seq = tokens[1:][:self.max_seq_len]
        pad_mask = [1] * len(input_seq)
        return torch.tensor(input_seq, dtype=torch.long), \
               torch.tensor(target_seq, dtype=torch.long), \
               torch.tensor(pad_mask, dtype=torch.long)

def collate_fn(batch):
    input_seqs, target_seqs, pad_masks = zip(*batch)
    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=vocab.word2idx["<pad>"])
    target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=vocab.word2idx["<pad>"])
    pad_masks = pad_sequence(pad_masks, batch_first=True, padding_value=0)
    return input_seqs, target_seqs, pad_masks

# Decoder-Only Transformer Model
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, pad_idx, max_seq_len):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(max_seq_len, embedding_dim), requires_grad=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seqs, pad_masks):
        seq_len = input_seqs.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :]
        embedded = self.embedding(input_seqs) + positional_encoding

        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(input_seqs.device)
        tgt_key_padding_mask = (pad_masks == 0)

        logits = self.decoder(
            tgt=embedded,
            memory=embedded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc(logits)

    def _generate_positional_encoding(self, max_seq_len, d_model):
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        positional_encoding = torch.zeros(max_seq_len, d_model)
        positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, 1::2] = torch.cos(pos * div_term)
        return positional_encoding.unsqueeze(0)

    def _generate_square_subsequent_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        return mask

# Training Loop
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for input_seqs, target_seqs, pad_masks in train_loader:
            logits = model(input_seqs, pad_masks)
            logits = logits.view(-1, logits.size(-1))
            target_seqs = target_seqs.view(-1)
            loss = criterion(logits, target_seqs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

# Perplexity Calculation
def calculate_sentence_perplexity(model, input_seqs, target_seqs, pad_masks, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(input_seqs, pad_masks)
        logits = logits.view(-1, logits.size(-1))
        target_seqs = target_seqs.view(-1)
        loss = criterion(logits, target_seqs)
        return torch.exp(loss).item()

# Generate Submission
def save_submission(model, test_loader, filename, criterion):
    model.eval()
    results = []

    with torch.no_grad():
        for idx, (input_seqs, target_seqs, pad_masks) in enumerate(test_loader):
            ppl = calculate_sentence_perplexity(model, input_seqs, target_seqs, pad_masks, criterion)
            results.append({'ID': idx, 'ppl': ppl})

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['ID', 'ppl'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Submission file saved as {filename}")

if __name__ == "__main__":
    train_data, val_data, test_data = load_data()
    train_data = preprocess_sentences(train_data)
    val_data = preprocess_sentences(val_data)
    test_data = preprocess_sentences(test_data)

    vocab = Vocabulary(VOCAB_SIZE)
    vocab.build_vocab(train_data)

    train_tokens = [vocab.encode(sentence) for sentence in train_data]
    val_tokens = [vocab.encode(sentence) for sentence in val_data]
    test_tokens = [vocab.encode(sentence) for sentence in test_data]

    train_dataset = PTBDataset(train_tokens, MAX_SEQ_LEN)
    val_dataset = PTBDataset(val_tokens, MAX_SEQ_LEN)
    test_dataset = PTBDataset(test_tokens, MAX_SEQ_LEN)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    model = DecoderOnlyTransformer(len(vocab.word2idx), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, vocab.word2idx["<pad>"], MAX_SEQ_LEN)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, optimizer, criterion, NUM_EPOCHS)
    save_submission(model, test_loader, "submission.csv", criterion)
