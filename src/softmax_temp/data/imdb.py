"""IMDB sentiment (binary). STUB: not implemented yet.

TODO for whoever picks this up:
  1. Load the dataset (e.g. `datasets.load_dataset("imdb")` from Hugging Face;
     torchtext is deprecated, so avoid it).
  2. Build a vocabulary from the TRAIN split only (cap at ~20-30k tokens), with
     <pad>=0 and <unk>=1, and tokenize (simple lowercase/regex is fine).
  3. Hold out `data.val_fraction` of train for validation using
     `split_indices(...)` so the split is shared across runs.
  4. Return datasets whose items are (LongTensor of token ids, int label), and
     put a `collate_fn` in `info` that pads to the batch max length with 0.
  5. Return info = {"num_classes": 2, "task": "text", "vocab_size": V,
                    "pad_idx": 0, "collate_fn": collate}

The GRU model in models/gru.py already expects this format.
"""
from __future__ import annotations


def build(cfg: dict):
    raise NotImplementedError("IMDB loader is a stub; see the TODO in data/imdb.py")
