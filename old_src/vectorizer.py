import json

class CharFrequencyVectorizer:
    def __init__(self, text=None, unk_token="<UNK>"):
        """
        Initialize the vectorizer. If text is provided, build the vocabulary.
        Otherwise, call build_vocab() later.

        Args:
            text (str, optional): Initial text to build vocab from.
            unk_token (str): The token used for unknown characters.
        """
        self.char2id = {}
        self.id2char = {}
        self.vocab_size = 0
        self.unk_token = unk_token
        
        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text):
        """
        Build the character-level vocabulary from the given text, including an unknown token.

        Args:
            text (str): The input text from which to build the vocabulary.
        """
        # Extract unique characters from the text
        unique_chars = sorted(set(text))

        # Insert the unknown token at the start
        all_chars = [self.unk_token] + unique_chars
        
        # Create mapping from char to ID
        self.char2id = {ch: idx for idx, ch in enumerate(all_chars)}
        self.id2char = {idx: ch for ch, idx in self.char2id.items()}
        
        self.vocab_size = len(all_chars)

    def transform(self, text):
        """
        Convert the given text into a frequency vector of size vocab_size.
        Unknown characters increment the count of the <UNK> token.

        Args:
            text (str): The text to convert into a frequency vector.

        Returns:
            list: A list of character frequencies of length vocab_size.
        """
        if not self.char2id:
            raise ValueError("Vocabulary not built. Please call build_vocab() first.")

        # Initialize frequency vector with zeros
        freq_vector = [0] * self.vocab_size

        unk_id = self.char2id[self.unk_token]

        # Count frequencies
        for ch in text:
            char_id = self.char2id.get(ch, unk_id)
            freq_vector[char_id] += 1
        
        return freq_vector

    def fit_transform(self, text):
        """
        Convenience method to first build the vocabulary from the given text,
        then return the frequency vector for the same text.

        Args:
            text (str): The text to build the vocabulary and transform.

        Returns:
            list: A list of character frequencies.
        """
        self.build_vocab(text)
        return self.transform(text)

    def save(self, file_path):
        """
        Save the vectorizer's state to a JSON file.

        Args:
            file_path (str): The path to the JSON file where the state will be saved.
        """
        state = {
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
            "char2id": self.char2id,
            "id2char": self.id2char
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def load(self, file_path):
        """
        Load the vectorizer's state from a JSON file into the current instance.

        Args:
            file_path (str): The path to the JSON file from which to load the state.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Update the current instance's state
        self.unk_token = state["unk_token"]
        self.vocab_size = state["vocab_size"]
        self.char2id = state["char2id"]
        self.id2char = state["id2char"]


# Example usage:
if __name__ == "__main__":
    sample_text = "Hello world!"
    vectorizer = CharFrequencyVectorizer(sample_text)
    freq_vector = vectorizer.transform("Hello")
    print("Original vectorizer char2id:", vectorizer.char2id)
    print("Transformed 'Hello':", freq_vector)

    # Save the vectorizer
    vectorizer.save("vectorizer_state.json")

    # Create a new vectorizer without building vocab
    new_vectorizer = CharFrequencyVectorizer()

    # Load the saved state into the new_vectorizer instance
    new_vectorizer.load("vectorizer_state.json")
    loaded_freq_vector = new_vectorizer.transform("Hello")
    print("Loaded vectorizer char2id:", new_vectorizer.char2id)
    print("Transformed 'Hello' with loaded vectorizer:", loaded_freq_vector)
