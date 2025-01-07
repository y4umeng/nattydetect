import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert to token IDs
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in text]

        # Pad or truncate
        if len(token_ids) < self.seq_len:
            token_ids = token_ids + [self.vocab["<pad>"]] * (self.seq_len - len(token_ids))
        else:
            token_ids = token_ids[:self.seq_len]

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)
