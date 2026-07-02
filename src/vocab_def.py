"""
vocab_def.py
============
Vocabulary class for mapping Ethereum counterparty addresses to integer token IDs.

This class mirrors the ``Vocab`` object produced by the BERT4ETH pre-processing
pipeline (``step27_prepare_bert4eth``), ensuring that pickled ``vocab.pkl`` files
can be deserialized correctly when ``__main__.Vocab`` is patched.

Usage::

    import pickle
    import sys
    import src.vocab_def as vocab_def
    sys.modules['__main__'].Vocab = vocab_def.Vocab

    vocab = pickle.load(open("data/bert4eth/vocab.pkl", "rb"))
"""


class Vocab:
    """Bidirectional mapping between address tokens and integer IDs.

    Special tokens:
        - ``[PAD]`` → 0 (padding, used as ``padding_idx`` in embeddings)
        - ``[UNK]`` → 1 (unknown addresses not seen during training)
    """

    def __init__(self) -> None:
        self.token2id: dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
        self.id2token: dict[int, str] = {0: "[PAD]", 1: "[UNK]"}

    def add(self, token: str) -> None:
        """Add a new token to the vocabulary if not already present.

        Args:
            token (str): The address string to register.
        """
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def get_id(self, token: str) -> int:
        """Look up the integer ID for a token.

        Args:
            token (str): The address string to look up.

        Returns:
            int: The token's ID, or 1 (``[UNK]``) if not found.
        """
        return self.token2id.get(token, 1)

    def __len__(self) -> int:
        return len(self.token2id)
