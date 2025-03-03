import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score


# Dataset Class
class POSDataset(Dataset):
    def __init__(self, file_path, token_vocab=None, tag_vocab=None, training=True):
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip()

        if training:
            assert 'IOB Slot tags' in data.columns, "The expected column 'IOB Slot tags' is not in the DataFrame."
        data['utterances'] = data['utterances'].apply(lambda x: x.split())

        if training:
            data['tags'] = data['IOB Slot tags'].apply(lambda x: x.split())
            data = data[data['utterances'].apply(len) > 0]
            data = data[data['utterances'].apply(len) == data['tags'].apply(len)]
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0}

            for utterance, tags in zip(data['utterances'], data['tags']):
                for token in utterance:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
                for tag in tags:
                    if tag not in self.tag_vocab:
                        self.tag_vocab[tag] = len(self.tag_vocab)
            self.idx_to_tag = {idx: tag for tag, idx in self.tag_vocab.items()}
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab
            self.idx_to_tag = {idx: tag for tag, idx in self.tag_vocab.items()}

        self.corpus_token_ids = [torch.tensor([self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in utterance]) for utterance in data['utterances']]

        if training:
            self.corpus_tag_ids = [torch.tensor([self.tag_vocab.get(tag, 0) for tag in tags]) for tags in data['tags']]
        else:
            self.corpus_tag_ids = None

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        if self.corpus_tag_ids is not None:
            return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]
        else:
            return self.corpus_token_ids[idx]


# Model Class
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        rnn_out, _ = self.lstm(embeddings)
        outputs = self.fc(rnn_out)

        return outputs


# Collate Function
def collate_fn(batch, token_vocab, tag_vocab):
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch]
    if len(token_ids) == 0:  # Handle empty batch
        return None, None
    sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=token_vocab['<PAD>'])
    tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=tag_vocab['<PAD>'])
    return sentences_padded, tags_padded


# Training and Evaluation Function
def train_and_evaluate(train_file, val_file, output_file, embedding_dim=100, hidden_dim=512, batch_size=32, num_epochs=25, learning_rate=0.001):
    train_dataset = POSDataset(train_file, training=True)
    val_dataset = POSDataset(val_file, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, train_dataset.token_vocab, train_dataset.tag_vocab))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, train_dataset.token_vocab, train_dataset.tag_vocab))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BiLSTMTagger(
        vocab_size=len(train_dataset.token_vocab),
        tagset_size=len(train_dataset.tag_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_vocab['<PAD>'])

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for token_ids, tag_ids in train_loader:
            if token_ids is None or tag_ids is None:
                continue  # Skip empty batch

            token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)

            optimizer.zero_grad()

            outputs = model(token_ids)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_tags = []

        with torch.no_grad():
            for token_ids, tag_ids in val_loader:
                if token_ids is None or tag_ids is None:
                    continue  # Skip empty batch

                token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)

                outputs = model(token_ids)
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
                total_val_loss += loss.item()

                predictions = outputs.argmax(dim=2)
                mask = tag_ids != train_dataset.tag_vocab['<PAD>']

                all_predictions.extend(predictions[mask].tolist())
                all_tags.extend(tag_ids[mask].tolist())

        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        f1 = f1_score(all_tags, all_predictions, average='weighted')

        print(f'{epoch + 1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')

    # Save Predictions
    idx_to_tag = {idx: tag for tag, idx in train_dataset.tag_vocab.items()}
    model.eval()
    predicted_tags = []

    with torch.no_grad():
        for token_ids in val_loader:
            if token_ids is None:
                continue  # Skip empty batch

            token_ids = token_ids.to(device)
            outputs = model(token_ids)
            predictions = outputs.argmax(dim=-1)
            for tokens, preds in zip(token_ids, predictions):
                predicted_seq = [idx_to_tag[idx.item()] for idx in preds if idx != train_dataset.tag_vocab['<PAD>']]
                predicted_tags.append(' '.join(predicted_seq))

    pd.DataFrame({'ID': range(1, len(predicted_tags) + 1), 'IOB Slot tags': predicted_tags}).to_csv(output_file, index=False)


# Main Execution
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_file> <val_file> <output_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    output_file = sys.argv[3]

    train_and_evaluate(train_file, val_file, output_file)
