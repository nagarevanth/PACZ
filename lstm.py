import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ─────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── File paths ───────────────────────────────────────────────────────────────────
sample_file = "datasets/sample_emails.csv"
genz_file   = "datasets/genz_emails_final_translated.csv'"

# ── Label mapping ────────────────────────────────────────────────────────────────
LABEL2IDX = {'Sender higher':0, 'Recipient higher':1, 'Similar level':2}

# ── Helper to auto-detect columns ─────────────────────────────────────────────────
def find_col(df, keywords):
    for col in df.columns:
        low = col.strip().lower()
        for kw in keywords:
            if kw in low:
                return col
    return None

# ── Vocabulary ───────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token2idx = {"<pad>":0, "<unk>":1}
        self.freqs = {}

    def build(self, texts):
        for txt in texts:
            for t in txt.split():
                self.freqs[t] = self.freqs.get(t, 0) + 1
        for tok, freq in self.freqs.items():
            if freq >= self.min_freq:
                self.token2idx.setdefault(tok, len(self.token2idx))

    def encode(self, text):
        return [ self.token2idx.get(t, 1) for t in text.split() ]

# ── Dataset & Collation ─────────────────────────────────────────────────────────
class EmailDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.seqs = [vocab.encode(t) for t in texts]
        self.labels = labels

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.seqs[i], self.labels[i]

def collate_fn(batch):
    seqs, labs = zip(*batch)
    lengths = [len(s) for s in seqs]
    maxlen = max(lengths)
    padded = [ s + [0]*(maxlen-len(s)) for s in seqs ]
    return (
      torch.tensor(padded, dtype=torch.long),
      torch.tensor(lengths, dtype=torch.long),
      torch.tensor(labs, dtype=torch.long)
    )

# ── Model ────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hid_dim, out_dim)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])

# ── Training & Evaluation ───────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, lengths, labels in loader:
        texts, lengths, labels = texts.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * texts.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for texts, lengths, labels in loader:
            texts, lengths = texts.to(DEVICE), lengths.to(DEVICE)
            logits = model(texts, lengths)
            batch_preds = logits.argmax(dim=1).cpu().tolist()
            preds.extend(batch_preds)
            labs.extend(labels.tolist())
    return preds, labs

# ── Main Pipeline ────────────────────────────────────────────────────────────────
# 1. Load & inspect sample
df = pd.read_csv(sample_file)
print("Sample columns:", df.columns.tolist())
content_col = find_col(df, ['content','body','text'])
label_col   = find_col(df, ['hierarchy'])
assert content_col and label_col, f"Columns not found in sample: {df.columns}"
df = df.dropna(subset=[content_col, label_col])
df['label_idx'] = df[label_col].map(LABEL2IDX)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label_idx']
)

# 2. Build vocab on TRAIN only
vocab = Vocabulary(min_freq=2)
vocab.build(train_df[content_col].tolist())

# 3. Prepare DataLoaders for sample
train_ds = EmailDataset(train_df[content_col].tolist(), train_df['label_idx'].tolist(), vocab)
test_ds  = EmailDataset(test_df [content_col].tolist(), test_df ['label_idx'].tolist(), vocab)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn)

# 4. Load & inspect Gen-Z
gdf = pd.read_csv(genz_file)
print("Gen-Z columns:", gdf.columns.tolist())
gen_col   = find_col(gdf, ['generated','genz','slang'])
label_col = find_col(gdf, ['hierarchy'])
assert gen_col and label_col, f"Columns not found in Gen-Z: {gdf.columns}"
gdf = gdf.dropna(subset=[gen_col, label_col])
gdf['label_idx'] = gdf[label_col].map(LABEL2IDX)
genz_ds = EmailDataset(gdf[gen_col].tolist(), gdf['label_idx'].tolist(), vocab)
genz_loader  = DataLoader(genz_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

# 5. Instantiate model, loss, optimizer
model = LSTMClassifier(
    vocab_size=len(vocab.token2idx),
    emb_dim=128, hid_dim=128, out_dim=3
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. Train
for epoch in range(1, 15):
    loss = train_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch}/5 — Loss: {loss:.4f}")

# 7. Evaluate on formal test split
preds_f, labs_f = evaluate(model, test_loader)
print("\n=== Formal Test Split ===")
print(classification_report(labs_f, preds_f, target_names=list(LABEL2IDX.keys()), digits=4))
print("Confusion Matrix:\n", confusion_matrix(labs_f, preds_f))

# 8. Evaluate on Gen-Z set
preds_g, labs_g = evaluate(model, genz_loader)
print("\n=== Gen-Z Set ===")
print(classification_report(labs_g, preds_g, target_names=list(LABEL2IDX.keys()), digits=4))
print("Confusion Matrix:\n", confusion_matrix(labs_g, preds_g))

# 9. (Optional) Save the model
torch.save(model.state_dict(), 'lstm2.pt')


