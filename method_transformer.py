"""Method 2 - an encoder-only transformer, trained from scratch (no pretraining).

Each letter is a token; a [CLS] token is put in front. The letters + positions go through 2 transformer-encoder layers
(self-attention), and the [CLS] vector is classified by a linear layer + softmax.

    python method_transformer.py    (or: python run.py transformer)
"""
import torch
import torch.nn as nn

from common import PAD, VOCAB, encode_chars, load_data, make_result
from trainer import DEVICE, fit_supervised, predict, set_seed


class CharTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.2, max_len=19):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):                      # x: (batch, 19) token ids, column 0 is [CLS]
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        h = self.norm(self.encoder(h, src_key_padding_mask=(x == PAD)))
        return self.head(self.drop(h[:, 0]))   # classify from the [CLS] position


def run(seed=0, strict=False, epochs=100, lr=1e-3, log=print):
    train, quiz, removed = load_data(strict=strict)
    set_seed(seed)
    encode = lambda words: torch.from_numpy(encode_chars(words)).to(DEVICE)
    model = CharTransformer().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log(f"  Transformer encoder: 2 layers, d_model 64, 4 heads ({params:,} weights), trained on {len(train)} words")
    fit_supervised(model, encode, train, epochs=epochs, lr=lr, seed=seed, log=log)
    pred, _ = predict(model, encode, [w for w, _ in quiz])
    return make_result("2. Transformer encoder, no pretraining", quiz, pred)


if __name__ == "__main__":
    import sys
    import run as runner
    runner.main(["transformer", *sys.argv[1:]])
