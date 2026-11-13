"""Small supervised-training helpers shared by the MLP and the from-scratch transformer scripts."""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PASTA, epoch_pairs, steps_per_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_supervised(model, encode, train, *, epochs, lr, bs=32, wd=0.05, ls=0.1, drop=0.1, seed=0, log=print):
    """Softmax / cross-entropy training. Each epoch is class-balanced and letter-dropped (see common.epoch_pairs)."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch(train, bs))
    for ep in range(epochs):
        model.train()
        loss_sum = correct = n = 0
        pairs = epoch_pairs(train, rng, drop)
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            y = torch.tensor([l for _, l in chunk], device=DEVICE)
            logits = model(encode([w for w, _ in chunk]))
            loss = F.cross_entropy(logits, y, label_smoothing=ls)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(chunk)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(chunk)
        if ep == 0 or (ep + 1) % max(1, epochs // 6) == 0:
            log(f"    epoch {ep + 1:3d}/{epochs}   loss {loss_sum / n:.3f}   train accuracy {correct / n:.1%}")


@torch.no_grad()
def predict(model, encode, words):
    """Returns (0/1 predictions, P(pasta)) from the softmax output."""
    model.eval()
    p = F.softmax(model(encode(words)), dim=-1)[:, PASTA]
    return (p > 0.5).long().cpu().numpy(), p.cpu().numpy()
