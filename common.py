"""
common.py

List of helpers that are shared by runs.

Framework-free helpers shared by the experiment: data, splits, features, metrics.

Nothing here imports torch, so it can be tested anywhere (see test_common.py).

Label convention everywhere: 0 = Italian surname (composer, painter, architect...), 1 = pasta shape.
"""
import math
import random
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
SURNAME, PASTA = 0, 1
CLASS_NAMES = ("surname", "pasta")

# --------------------------------------------------------------------
# Data functions
# --------------------------------------------------------------------

def read_words(path):
    return [w.strip() for w in open(path, encoding="utf-8") if w.strip()]


def load_data(strict=False, similarity=0.8):
    """Return (train_pairs, quiz_pairs, removed_words).

    train_pairs / quiz_pairs are lists of (word, label). The 40 quiz words are guaranteed not to be in
    train_pairs. With strict=True, every training word whose spelling is >= `similarity` similar
    (difflib ratio) to ANY quiz word is also removed, so near-variants (Bellini/Bellino, ziti/zite...)
    cannot leak the answer.
    """
    surnames = read_words(DATA_DIR / "surnames.txt")
    pastas = read_words(DATA_DIR / "pastas.txt")
    quiz = []
    for line in open(DATA_DIR / "quiz40.tsv", encoding="utf-8"):
        if line.strip():
            w, l = line.rstrip("\n").split("\t")
            quiz.append((w, int(l)))
    train = [(w, SURNAME) for w in surnames] + [(w, PASTA) for w in pastas]
    quiz_words = {w for w, _ in quiz}
    assert not quiz_words & {w for w, _ in train}, "quiz word found in training data"
    assert len({w for w, _ in train}) == len(train), "duplicate word in training data"
    removed = []
    if strict:
        keep = []
        for w, l in train:
            if any(SequenceMatcher(None, w, q).ratio() >= similarity for q in quiz_words):
                removed.append(w)
            else:
                keep.append((w, l))
        train = keep
    return train, quiz, removed


def split_val(pairs, frac, seed):
    """Hold out ~frac of the pairs as a validation set, grouped by the first 3 letters so that
    spelling-siblings (Ricci/Riccio/Riccardi) never straddle the split."""
    rng = random.Random(seed)
    size = Counter(w[:3] for w, _ in pairs)
    groups = sorted(size)
    rng.shuffle(groups)
    target, taken, val_groups = frac * len(pairs), 0, set()
    for g in groups:
        if taken >= target:
            break
        val_groups.add(g)
        taken += size[g]
    train = [p for p in pairs if p[0][:3] not in val_groups]
    val = [p for p in pairs if p[0][:3] in val_groups]
    assert {l for _, l in val} == {SURNAME, PASTA}, "validation split is missing a class"
    return train, val


def drop_letters(word, rng, p):
    """Augmentation: delete each letter with probability p (keeps >= 3 letters)."""
    if p <= 0 or len(word) <= 4:
        return word
    kept = [c for c in word if rng.random() >= p]
    return "".join(kept) if len(kept) >= 3 else word


def epoch_pairs(train, rng, drop=0.0):
    """One class-balanced, shuffled, augmented epoch: all words of the smaller class plus an equal-sized
    random sample of the larger class (a different sample every epoch, so nothing is thrown away)."""
    by = {SURNAME: [p for p in train if p[1] == SURNAME], PASTA: [p for p in train if p[1] == PASTA]}
    n = min(len(by[SURNAME]), len(by[PASTA]))
    pairs = rng.sample(by[SURNAME], n) + rng.sample(by[PASTA], n)
    rng.shuffle(pairs)
    return [(drop_letters(w, rng, drop), l) for w, l in pairs]


def steps_per_epoch(train, batch_size):
    n = 2 * min(sum(1 for _, l in train if l == SURNAME), sum(1 for _, l in train if l == PASTA))
    return math.ceil(n / batch_size)

# ------------------------------------------------------------ features ----

def ngrams(word, n_max=3):
    s = "^" + word + "$"
    return [s[i:i + n] for n in range(1, n_max + 1) for i in range(len(s) - n + 1)]


class NgramVocab:
    """Binary bag of character 1-3-grams (with ^ and $ word-boundary markers) for the MLP.
    The vocabulary is built from the training words only."""

    def __init__(self, words, min_count=2):
        c = Counter(g for w in words for g in set(ngrams(w)))
        self.index = {g: i for i, g in enumerate(sorted(g for g, k in c.items() if k >= min_count))}

    def __len__(self):
        return len(self.index)

    def transform(self, words):
        X = np.zeros((len(words), len(self.index)), dtype=np.float32)
        for i, w in enumerate(words):
            for g in ngrams(w):
                j = self.index.get(g)
                if j is not None:
                    X[i, j] = 1.0
        return X


# Token ids for the transformer
PAD, UNK, CLS, MASK, LETTER0 = 0, 1, 2, 3, 4
VOCAB = LETTER0 + 26
MAX_LEN = 18  # longest word in the data is 17 letters


def encode_chars(words, max_len=MAX_LEN):
    X = np.zeros((len(words), max_len + 1), dtype=np.int64)  # column 0 is [CLS]
    for i, w in enumerate(words):
        X[i, 0] = CLS
        for j, ch in enumerate(w[:max_len]):
            X[i, j + 1] = LETTER0 + ord(ch) - 97 if "a" <= ch <= "z" else UNK
    return X

# ------------------------------------------------------------- metrics ----

def accuracy(pred, y):
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def balanced_accuracy(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in (SURNAME, PASTA)]))

# ---------------------------------------------------- hyper-parameter search ----

_ARCH = dict(d_model=[64, 128], layers=[2, 3], dropout=[0.2, 0.3])
SEARCH_SPACE = {
    "mlp": dict(hidden=[(64, 32), (128, 64), (256, 64)], dropout=[0.3, 0.5], lr=[1e-3, 3e-3], epochs=[60, 120]),
    "tf_scratch": dict(**_ARCH, lr=[3e-4, 1e-3], epochs=[60, 120]),
    "tf_pre_sl": dict(**_ARCH, lr=[3e-4, 1e-3], epochs=[30, 60], pretrain_epochs=[100, 200]),
    "tf_pre_rl": dict(**_ARCH, rl_lr=[5e-4, 1e-3, 2e-3], rl_epochs=[100, 200], entropy=[0.01, 0.03],
                      pretrain_epochs=[100, 200]),
    "tf_pre_sl_rl": dict(**_ARCH, lr=[3e-4, 1e-3], epochs=[30, 60], polish_lr=[1e-4, 2e-4, 5e-4],
                         polish_epochs=[10, 20, 40], pretrain_epochs=[100, 200]),
}
DEFAULTS = {
    "mlp": dict(hidden=(64, 32), dropout=0.3, lr=1e-3, epochs=60),
    "tf_scratch": dict(d_model=64, layers=2, dropout=0.2, lr=1e-3, epochs=60),
    "tf_pre_sl": dict(d_model=64, layers=2, dropout=0.2, lr=1e-3, epochs=60, pretrain_epochs=100),
    "tf_pre_rl": dict(d_model=64, layers=2, dropout=0.2, rl_lr=1e-3, rl_epochs=100, entropy=0.01, pretrain_epochs=100),
    "tf_pre_sl_rl": dict(d_model=64, layers=2, dropout=0.2, lr=1e-3, epochs=60, polish_lr=2e-4, polish_epochs=20,
                         pretrain_epochs=100),
}


def sample_configs(key, n, seed=0):
    """The default config first, then n-1 distinct random draws from the key's search space."""
    rng = random.Random(seed)
    space, out = SEARCH_SPACE[key], [dict(DEFAULTS[key])]
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        cfg = {k: rng.choice(v) for k, v in space.items()}
        if cfg not in out:
            out.append(cfg)
    return out

# ------------------------------------------- quiz results & LLM answer helpers ----

def make_result(name, quiz, pred, **extra):
    """Uniform result record used by run.py. pred: one 0/1 per quiz word (-1 = no valid answer)."""
    pred = [int(p) for p in pred]
    truth = [l for _, l in quiz]
    n = sum(p == t for p, t in zip(pred, truth))
    return dict(name=name, quiz_acc=n / len(quiz), n_correct=n, n=len(quiz), quiz_pred=pred, **extra)


LLM_SYSTEM = ("You classify words. Reply with exactly one word: 'surname' if the word is an Italian family name, "
              "or 'pasta' if it is the name of a pasta shape.")
# well-known example words, shown to the model as earlier turns of the conversation (none is a quiz word)
LLM_SHOTS = [("Rossi", "surname"), ("penne", "pasta"), ("Verdi", "surname"), ("rigatoni", "pasta"),
             ("Vivaldi", "surname"), ("farfalle", "pasta"), ("Bianchi", "surname"), ("fusilli", "pasta")]


def llm_messages(word):
    """Chat messages for one word: instruction, 8 worked examples, then the question."""
    msgs = [{"role": "system", "content": LLM_SYSTEM}]
    for w, a in LLM_SHOTS:
        msgs += [{"role": "user", "content": w}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": word.capitalize()}]


def llm_plain(word):
    """Same thing as plain text, for models that have no chat template."""
    shots = "\n".join(f"{w}: {a}" for w, a in LLM_SHOTS)
    return f"{LLM_SYSTEM}\n\n{shots}\n{word.capitalize()}:"


def parse_label(text):
    """Turn a generated answer into a label: PASTA, SURNAME, or None if it said neither."""
    t = text.lower()
    i_p = t.find("pasta")
    hits = [i for i in (t.find("surname"), t.find("composer")) if i >= 0]
    i_s = min(hits) if hits else -1
    if i_p >= 0 and (i_s < 0 or i_p < i_s):
        return PASTA
    return SURNAME if i_s >= 0 else None


def answer_reward(label, truth):
    """RL reward: 1 for the right category, 0 for the wrong one, -0.5 for an answer that is neither."""
    return -0.5 if label is None else (1.0 if label == truth else 0.0)
