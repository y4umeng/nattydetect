import json
import math
import os
from random import shuffle
from datasets import load_dataset
from vectorizer import CharFrequencyVectorizer
from src.naive_bayes import NaiveBayesBinaryClassifier
def main():
    # Load datasets
    nl_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    nl_texts = nl_dataset["text"]
    nl_texts = [t for t in nl_texts if t.strip()]

    code_dataset = load_dataset("code_search_net", split="train")
    code_texts = code_dataset["func_code_string"]
    code_texts = [t for t in code_texts if t and t.strip()]

    print(f"Code texts: {len(code_texts)}")

    shuffle(code_texts)

    len_data = min(100000, min(len(nl_texts), len(code_texts)))
    train_cut = 0.75
    cutoff = (int) (len_data*train_cut)
    
    nl_texts = nl_texts[:len_data]
    code_texts = code_texts[:len_data]

    print(f"nl_texts: {len(nl_texts)}")
    print(f"code_texts: {len(code_texts)}")
    # Split into train/validation
    nl_train = nl_texts[:cutoff]
    nl_val = nl_texts[cutoff:len_data]

    code_train = code_texts[:cutoff]
    code_val = code_texts[cutoff:len_data]

    # Combine train and val sets
    X_train_text = nl_train + code_train
    y_train = [0]*len(nl_train) + [1]*len(code_train)

    print(f"y_train: {len(y_train)}")

    X_val_text = nl_val + code_val
    y_val = [0]*len(nl_val) + [1]*len(code_val)

    print(f"y_val: {len(y_val)}")

    # Build vectorizer on training data only
    combined_train_text = "".join(X_train_text)
    vectorizer = CharFrequencyVectorizer()
    vectorizer.build_vocab(combined_train_text)

    X_train = [vectorizer.transform(txt) for txt in X_train_text]
    X_val = [vectorizer.transform(txt) for txt in X_val_text]

    # Train NB model
    clf = NaiveBayesBinaryClassifier(alpha=1.0)
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    correct = 0
    preds = 0
    for x, label in zip(X_val, y_val):
        pred = clf.predict(x)
        preds += pred
        if pred == label:
            correct += 1
    accuracy = correct / len(y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"True predictions: {preds}")
    print(f"Validation data: {len(y_val)}")
    print(f"Sum y_val: {sum(y_val)}")
    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    vectorizer.save("models/vectorizer.json")
    clf.save("models/naive_bayes.json")

    print("Training complete. Model and vectorizer saved.")


if __name__ == "__main__":
    main()