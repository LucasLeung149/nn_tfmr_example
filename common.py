"""
common.py

List of helpers that are shared by the experiment.

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

# -----------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------

def read_words(path):
    """
    Read non-empty words from a UTF-8 text file.

    Reads a text file containing one word per line.
    Leading and trailing whitespace is removed, and
    empty or whitespace-only lines are ignored.

    Args:
        path (str | Path): Path to the text file.

    Returns:
        list[str]: List of words in their original order,
            excluding empty lines.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
        OSError: If the file cannot be opened or read.
    """
    return [w.strip() for w in open(path, encoding="utf-8") if w.strip()]



def load_data(strict=False, similarity=0.8):
    """Load the labeled training words and held-out quiz dataset.

    Reads Italian surnames and pasta names from text files and loads
    the 40-word evaluation quiz from a TSV file.

    Quiz words are never included in the training dataset.

    When strict mode is enabled, training words that are too similar to any quiz word are also excluded. 
    This reduces the risk of evaluating the model on near-duplicates of words it encountered during training.

    Similarity is calculated using difflib.SequenceMatcher.

    Args:
        strict (bool): If True, remove training words whose spelling similarity 
            to any quiz word is greater than or equal to the similarity threshold.
            Defaults to False.

        similarity (float): Similarity threshold used in strict mode. 
        Defaults to 0.8.

    Returns:
        tuple:
            train_pairs (list[tuple[str, int]]):
                Training words paired with their class labels.

            quiz_pairs (list[tuple[str, int]]):
                Held-out quiz words and their correct labels.

            removed_words (list[str]):
                Training words excluded by strict filtering.
                Empty when strict=False.

    Raises:
        AssertionError: If a quiz word appears in the
            training dataset or if training words are duplicated.

    Notes:
        Class labels:
            0 = Italian surname
            1 = Pasta shape

        Strict filtering compares the original spelling
        using a case-sensitive similarity measure.
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

# -----------------------------------------------------------------
# Features and Embeddings
# -----------------------------------------------------------------

def ngrams(word, n_max=3):
    s = "^" + word + "$"
    return [s[i:i + n] for n in range(1, n_max + 1) for i in range(len(s) - n + 1)]



class NgramVocab:
    """
    Build a vocabulary of character n-gram features for the MLP.

    Extracts character 1-, 2-, and 3-grams from training words,
    including start (^) and end ($) boundary markers.

    Only n-grams appearing in at least min_count distinct
    training words are included.

    Each word is represented as a binary feature vector:
        1 = the n-gram appears in the word
        0 = the n-gram does not appear

    The representation ignores how many times an n-gram occurs.

    Attributes:
        index (dict[str, int]): Mapping from each retained
            n-gram to its feature column index.
    """

    def __init__(self, words, min_count=2):
        """Construct the n-gram vocabulary.

        Args:
            words (list[str]): Training words used to build
                the vocabulary.
            min_count (int): Minimum number of distinct words
                containing an n-gram for it to be retained.
        """
        c = Counter(g for w in words for g in set(ngrams(w)))
        self.index = {
            g: i
            for i, g in enumerate(
                sorted(g for g, k in c.items() if k >= min_count)
            )
        }

    def __len__(self):
        """Return the number of retained n-gram features."""
        return len(self.index)

    def transform(self, words):
        """Encode words as binary n-gram feature vectors.

        Args:
            words (list[str]): Words to encode using the
                existing vocabulary.

        Returns:
            np.ndarray: Float32 array of shape
                (number_of_words, vocabulary_size).

        Notes:
            Unknown n-grams are ignored.
            The vocabulary is not modified during encoding.
        """
        X = np.zeros(
            (len(words), len(self.index)),
            dtype=np.float32
        )
        for i, w in enumerate(words):
            for g in ngrams(w):
                j = self.index.get(g)
                if j is not None:
                    X[i, j] = 1.0
        return X

# Token ids for the transformer
# Reserved token IDs: PAD (0) for padding, UNK (1) for unknown characters,
# CLS (2) for sequence classification, MASK (3) for masked-token tasks,
# and LETTER0 (4) as the starting ID for lowercase letters a-z (IDs 4-29).
PAD, UNK, CLS, MASK, LETTER0 = 0, 1, 2, 3, 4
VOCAB = LETTER0 + 26
MAX_LEN = 18  # longest word in the data is 17 letters


def encode_chars(words, max_len=MAX_LEN):
    """
    Convert words into fixed-length sequences of character token IDs.

    Each word is encoded as a sequence of integers for use as input
    to the Transformer model. The first position contains the special
    CLS token, followed by the encoded characters of the word.

    Lowercase letters (a-z) are mapped to token IDs 4-29.
    Characters outside this range are assigned the UNK token.
    Words shorter than max_len are padded with zeros (PAD),
    while longer words are truncated to max_len characters.

    Args:
        words (list[str]): List of words to encode.
        max_len (int): Maximum number of characters per word,
            excluding the CLS token. Defaults to MAX_LEN (18).

    Returns:
        np.ndarray: A 2D integer array of shape
            (len(words), max_len + 1), with dtype int64.
            Each row represents one encoded word, where:
                - Column 0 contains the CLS token (ID 2).
                - Remaining columns contain character token IDs.
                - Unused positions contain PAD tokens (ID 0).
    """

    X = np.zeros((len(words), max_len + 1), dtype=np.int64)  # column 0 is [CLS]
    for i, w in enumerate(words):
        X[i, 0] = CLS
        for j, ch in enumerate(w[:max_len]):
            X[i, j + 1] = LETTER0 + ord(ch) - 97 if "a" <= ch <= "z" else UNK
    return X

# -----------------------------------------------------------------
# Metrics as helpers
# -----------------------------------------------------------------

def accuracy(pred, y):
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def balanced_accuracy(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in (SURNAME, PASTA)]))

# -----------------------------------------------------------------
# Hyperparameter search
# -----------------------------------------------------------------

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

# -----------------------------------------------------------------
# Quiz Results, LLM helpers
# -----------------------------------------------------------------


def make_result(name, quiz, pred, **extra):
    """
    Create a standardized result record for evaluating a model.

    Compares the model's predicted labels against the correct quiz
    labels and calculates the number and proportion of correct
    predictions.

    Args:
        name (str): Name of the model or experiment.
        quiz (list[tuple[str, int]]): Quiz words paired with their
            correct labels (0 = surname, 1 = pasta).
        pred (list[int]): Predicted labels in the same order as
            the quiz words. A value of -1 indicates an invalid
            or unrecognized prediction.
        **extra: Additional information to include in the
            result dictionary, such as before-RL accuracy.

    Returns:
        dict: Dictionary containing:
            name (str): Model or experiment name.
            quiz_acc (float): Fraction of correctly classified
                quiz words, between 0 and 1.
            n_correct (int): Number of correct predictions.
            n (int): Total number of quiz words.
            quiz_pred (list[int]): Model predictions.
            Additional fields supplied through **extra.

    Raises:
        ValueError: If the quiz is empty, prediction count does
            not match the quiz size, or predictions contain
            labels other than -1, 0, or 1.
    """
    pred = [int(p) for p in pred]

    if not quiz:
        raise ValueError("Quiz must contain at least one word.")

    if len(pred) != len(quiz):
        raise ValueError("Number of predictions must match the quiz size.")

    if any(p not in (-1, SURNAME, PASTA) for p in pred):
        raise ValueError("Predictions must be -1, 0, or 1.")

    truth = [label for _, label in quiz]
    n_correct = sum(p == t for p, t in zip(pred, truth))
    return dict(name=name, quiz_acc=n_correct / len(quiz), n_correct=n_correct, n=len(quiz), quiz_pred=pred, **extra)


LLM_SYSTEM = ("You classify words. Reply with exactly one word: 'surname' if the word is an Italian family name, "
              "or 'pasta' if it is the name of a pasta shape.")
# well-known example words, shown to the model as earlier turns of the conversation (none is a quiz word)
LLM_SHOTS = [("Rossi", "surname"), ("penne", "pasta"), ("Verdi", "surname"), ("rigatoni", "pasta"),
             ("Vivaldi", "surname"), ("farfalle", "pasta"), ("Bianchi", "surname"), ("fusilli", "pasta")]



def llm_messages(word):
    """Construct a few-shot chat prompt for classifying a word.

    Creates a conversation containing a system instruction, eight
    labeled examples of Italian surnames and pasta shapes, and
    the target word as the final user message.

    Args:
        word (str): Word to classify as a surname or pasta shape.

    Returns:
        list[dict]: Chat messages containing:
            - A system message defining the classification task.
            - Eight user-assistant example pairs from LLM_SHOTS.
            - A final user message containing the target word.

    Notes:
        The target word is capitalized using str.capitalize().
        The output is intended for models supporting chat templates.
    """
    msgs = [{"role": "system", "content": LLM_SYSTEM}]
    for w, a in LLM_SHOTS:
        msgs += [{"role": "user", "content": w}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": word.capitalize()}]


def llm_plain(word):
    """Construct a plain-text few-shot prompt for word classification.

    Formats the system instruction and eight labeled examples as
    a single string, followed by the target word and a colon.
    This allows language models without chat templates to perform
    the same surname-versus-pasta classification task.

    Args:
        word (str): Word to classify as a surname or pasta shape.

    Returns:
        str: A formatted prompt containing the classification
            instruction, labeled examples, and target word.

    Notes:
        The target word is capitalized using str.capitalize().
        The trailing colon encourages the model to generate
        the corresponding class label.
    """
    shots = "\n".join(f"{w}: {a}" for w, a in LLM_SHOTS)
    return f"{LLM_SYSTEM}\n\n{shots}\n{word.capitalize()}:"


def parse_label(text):
    """Extract a classification label from an LLM-generated response.

    Searches the generated text for category keywords and converts
    the response into the corresponding numerical class label.

    The matching is case-insensitive:
        - "pasta" indicates the PASTA class (1).
        - "surname" or "composer" indicates the SURNAME class (0).
        - If both categories appear, the earliest matching
          keyword determines the predicted class.
        - If no keyword is found, the response is invalid.

    Args:
        text (str): Text generated by the language model.

    Returns:
        int | None: Predicted classification label:
            0 (SURNAME): Italian surname.
            1 (PASTA): Pasta shape.
            None: No recognized category found.

    Notes:
        Keywords are matched as substrings, not complete words.
        Therefore, the function may recognize keywords embedded
        within longer words or sentences.
    """
    t = text.lower()
    i_p = t.find("pasta")
    hits = [i for i in (t.find("surname"), t.find("composer")) if i >= 0]
    i_s = min(hits) if hits else -1
    if i_p >= 0 and (i_s < 0 or i_p < i_s):
        return PASTA
    return SURNAME if i_s >= 0 else None




def answer_reward(label, truth):
    """Calculate the reinforcement learning reward for an LLM answer.

    Compares the category extracted from the generated text
    against the correct category.

    The reward function is deterministic and does not require
    a learned reward model.

    Args:
        label (int | None): Predicted class returned by
            parse_label(). None indicates an invalid answer.

        truth (int): Correct class label.
            0 = Italian surname
            1 = Pasta shape

    Returns:
        float: Reward assigned to the generated answer.

            1.0: Correct classification.
            0.0: Incorrect classification.
           -0.5: Invalid or unrecognized answer.

    Examples:
        >>> answer_reward(SURNAME, SURNAME)
        1.0

        >>> answer_reward(PASTA, SURNAME)
        0.0

        >>> answer_reward(None, SURNAME)
        -0.5

    Notes:
        This reward is used by the group-relative
        reinforcement learning training loop.

        The reward is not differentiable. Instead, it is
        used to weight the log-probability of generated
        responses in the policy-gradient objective.
    """
    return -0.5 if label is None else (1.0 if label == truth else 0.0)

