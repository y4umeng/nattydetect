from datasets import load_dataset
import random
from collections import Counter
import os

def load_data():
    # Load datasets
    nl_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    code_dataset = load_dataset("code_search_net", split="train")

    # Extract text fields
    # wikitext entries have 'text'
    nl_texts = [x['text'] for x in nl_dataset if x['text'].strip() != '']
    # code_search_net entries have a 'code' field which is code text
    code_texts = [x['func_code_string'] for x in code_dataset if x['func_code_string'].strip() != '']

    # For demonstration, let's sample smaller subsets:
    nl_sample = random.sample(nl_texts, 5000)    # sample 5k NL lines
    code_sample = random.sample(code_texts, 5000) # sample 5k code lines

    # Label: NL=0, Code=1
    texts = nl_sample + code_sample
    labels = [0]*len(nl_sample) + [1]*len(code_sample)

    # Shuffle combined data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def tokenize(text):
    # Simple whitespace tokenizer
    return text.strip().split()


def build_vocab(tokenized_texts, vocab_size=5000):
    # Count all tokens
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    # Most common tokens
    most_common = counter.most_common(vocab_size-2)  # leave space for <pad>, <unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, (token, _) in enumerate(most_common, start=2):
        vocab[token] = i
    return vocab


def train_val_split(texts, labels, val_ratio=0.1):
    total = len(texts)
    val_size = int(total*val_ratio)
    return (texts[val_size:], labels[val_size:]), (texts[:val_size], labels[:val_size])


def ensure_reproducibility(seed=42):
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
