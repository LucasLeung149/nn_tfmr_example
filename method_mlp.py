"""
method_mlp.py

Method 1 - a plain 3-layer MLP with a softmax output, on character n-gram features.

The embedding uses the Ngram vocab for embedding, with ^ and $ acting as the begin and end markers/
The MLP set up is as follows

    features -> 64 -> 32 -> 2 (softmax) for class 

Classes: 
    0 - surname
    1 - pasta

To run in terminal:

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
            # fill it in!
            # Layer 1 -- linear, ReLU and dropout layer

            # Layer 2 -- linear, ReLU and dropout

            # Layer 3 -- map to classes

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
