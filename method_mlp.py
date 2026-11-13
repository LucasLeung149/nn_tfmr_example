"""Method 1 - a plain 3-layer MLP with a softmax output, on character n-gram features.

The word is turned into a 0/1 vector saying which letter 1-, 2- and 3-grams it contains (with ^ and $ marking the
start and end), then goes through  features -> 64 -> 32 -> 2  and a softmax. No sequence model, no pretraining.

    python method_mlp.py            (or: python run.py mlp)
"""
import torch
import torch.nn as nn

from common import NgramVocab, load_data, make_result
from trainer import DEVICE, fit_supervised, predict, set_seed


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(64, 32), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1 -- linear, ReLU and dropout layer
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 2)
            )

    def forward(self, x):
        return self.net(x)


def run(seed=0, strict=False, epochs=100, lr=1e-3, log=print):
    train, quiz, removed = load_data(strict=strict)
    # set the seed from the training
    set_seed(seed)

    # Embedding
    # for each of the training data we embed the word using ngram
    vocab = NgramVocab([w for w, _ in train])
    encode = lambda words: torch.from_numpy(vocab.transform(words)).to(DEVICE)

    # Create MLP network
    model = MLP(len(vocab)).to(DEVICE)
    # number of parameters
    params = sum(p.numel() for p in model.parameters())
    # printing stuff
    log(f"  MLP: {len(vocab)} n-gram features -> 64 -> 32 -> 2  ({params:,} weights), trained on {len(train)} words")

    # Training
    # run supevised running
    fit_supervised(model, encode, train, epochs=epochs, lr=lr, seed=seed, log=log)

    # Running
    # test prdiction using the quiz words
    pred, _ = predict(model, encode, [w for w, _ in quiz])
    return make_result("1. MLP (3 layers, softmax)", quiz, pred)


# if this is the main script, we run the runner lines in the run file.
if __name__ == "__main__":
    import sys
    import run as runner
    runner.main(["mlp", *sys.argv[1:]])
