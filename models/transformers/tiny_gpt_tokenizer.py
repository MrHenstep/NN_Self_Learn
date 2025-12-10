class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)
    