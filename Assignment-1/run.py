import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path, embedding_dim):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Convert words to GloVe embeddings
def sentence_to_avg_embedding(sentence, embeddings_index, embedding_dim):
    words = sentence.split()
    word_embeddings = [embeddings_index.get(word, np.zeros(embedding_dim)) for word in words]
    return np.mean(word_embeddings, axis=0)

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        return torch.sigmoid(self.fc5(x))

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training with Early Stopping
def train_model_with_early_stopping(model, X_train, y_train, X_val, y_val, criterion, optimizer, scheduler, num_epochs=50, batch_size=32):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    train_losses, val_losses, val_f1_scores = [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            labels = y_train[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(X_train))

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_preds = (val_outputs > 0.5).float()
            val_f1 = f1_score(y_val, val_preds, average="micro")
        scheduler.step()
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(X_train):.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses, val_f1_scores

def main(train_data_path, test_data_path, output_file_path):
    # Load train and test datasets
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # Preprocess the data
    train_data['CORE RELATIONS'] = train_data['CORE RELATIONS'].apply(lambda x: x.split())
    
    # Load GloVe embeddings
    glove_file = 'glove.6B.300d.txt'  # Ensure this file is in the same directory
    embedding_dim = 300
    embeddings_index = load_glove_embeddings(glove_file, embedding_dim)

    # Prepare embeddings for train and test data
    X_glove = np.array([sentence_to_avg_embedding(utterance, embeddings_index, embedding_dim) for utterance in train_data['UTTERANCES']])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train_data['CORE RELATIONS'])

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_glove, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Model Initialization
    input_dim = embedding_dim
    output_dim = y_train.shape[1]
    model = MLP(input_dim, output_dim)

    # Define Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    # Train model
    train_model_with_early_stopping(model, X_train, y_train, X_val, y_val, criterion, optimizer, scheduler)

    # Prepare test data for predictions
    X_test_glove = np.array([sentence_to_avg_embedding(utterance, embeddings_index, embedding_dim) for utterance in test_data['UTTERANCES']])
    X_test_glove = torch.tensor(X_test_glove, dtype=torch.float32)

    # Predict using the trained model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_glove)
        test_preds = (test_outputs > 0.5).float()

    # Convert predictions back to labels
    predicted_relations = mlb.inverse_transform(test_preds.numpy())
    test_data['CORE RELATIONS'] = [' '.join(rel) for rel in predicted_relations]
    
    # Save predictions to CSV
    submission = test_data[['ID', 'CORE RELATIONS']]
    submission.to_csv(output_file_path, index=False)
    print(f"Submission file saved as {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_data> <test_data> <output_file>")
    else:
        train_data_path = sys.argv[1]
        test_data_path = sys.argv[2]
        output_file_path = sys.argv[3]
        main(train_data_path, test_data_path, output_file_path)
