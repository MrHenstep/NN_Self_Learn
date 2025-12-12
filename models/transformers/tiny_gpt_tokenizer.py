from tokenizers import ByteLevelBPETokenizer
import os

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

class ByteBPETokenizerWrapper:
    """
    Minimal wrapper around Hugging Face ByteLevelBPETokenizer
    providing encode/decode + vocab_size, matching your CharTokenizer interface.
    """
    def __init__(self, tokenizer: ByteLevelBPETokenizer):
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, s: str):
        return self._tok.encode(s).ids

    def decode(self, ids):
        return self._tok.decode(ids)


def build_or_load_bytebpe(text_file: str,
                          vocab_size: int = 8000,
                          min_frequency: int = 2,
                          cache_dir: str = "bpe_tok"):
    """
    Train Byte-level BPE on `text_file` and save vocab/merges to `cache_dir`.
    If files exist, load instead.
    """
    os.makedirs(cache_dir, exist_ok=True)
    vocab_path = os.path.join(cache_dir, "vocab.json")
    merges_path = os.path.join(cache_dir, "merges.txt")

    if os.path.isfile(vocab_path) and os.path.isfile(merges_path):
        bpe = ByteLevelBPETokenizer(vocab_path, merges_path)
    else:
        if not text_file or not os.path.isfile(text_file):
            raise ValueError("Byte-level BPE training requires --text_file pointing to a real .txt file.")
        bpe = ByteLevelBPETokenizer()
        bpe.train(
            files=[text_file],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>", "<bos>", "<eos>"],  # optional but handy
        )
        bpe.save_model(cache_dir)

    return ByteBPETokenizerWrapper(bpe)
