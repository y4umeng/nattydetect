import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, tokenize, build_vocab, train_val_split, ensure_reproducibility
from dataset import TextClassificationDataset
from model import DNNLinearCombinedModel
import wandb

def main():
    # Initialize wandb
    wandb.init(project="nattydetect", config={
        "batch_size": 64,
        "epochs": 5,
        "learning_rate": 0.001,
        "vocab_size": 5000,
        "sequence_length": 500
    })
    config = wandb.config

    ensure_reproducibility()

    # 1. Load and prepare data
    print("Loading data...")
    texts, labels = load_data()
    tokenized_texts = [tokenize(t) for t in texts]

    # 2. Build vocabulary
    print("Building vocab.")
    vocab = build_vocab(tokenized_texts, vocab_size=config.vocab_size)

    # 3. Split into train/val
    (train_texts, train_labels), (val_texts, val_labels) = train_val_split(tokenized_texts, labels)

    # 4. Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, seq_len=config.sequence_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, vocab, seq_len=config.sequence_length)

    print(f"Train set length: {len(train_dataset)}")

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 6. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNLinearCombinedModel(vocab_size=len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SparseAdam(model.parameters(), lr=config.learning_rate)

    # Watch the model with wandb
    wandb.watch(model, log="all")

    
    # 7. Training loop
    print("Training start.")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch_tokens, batch_labels in train_loader:
            batch_tokens, batch_labels = batch_tokens.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_tokens)
            loss = criterion(logits.squeeze(), batch_labels.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_tokens, batch_labels in val_loader:
                batch_tokens, batch_labels = batch_tokens.to(device), batch_labels.to(device)
                logits = model(batch_tokens)
                preds = torch.sigmoid(logits)
                predicted = (preds > 0.5).float()
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)

        val_acc = correct / total

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc
        })

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model
    model_path = "binary_classifier.pt"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

if __name__ == "__main__":
    main()
