"""Run one method (or all three) and test it on the 40 quiz words.

    python run.py mlp                  # 1. 3-layer MLP
    python run.py transformer          # 2. encoder-only transformer, no pretraining
    python run.py pretrained           # 3. pretrained language model + RL
    python run.py all                  # all three, then a comparison table and chart

Useful options:  --seed N   --strict (also drop training words that look like a quiz word)
                 --epochs N (mlp / transformer)   --model NAME --steps N --batch N --group N --no-lora (pretrained)
The scripts can also be run directly:  python method_mlp.py  etc.
"""
import argparse
import importlib
import inspect
import json
import time
from pathlib import Path

from common import CLASS_NAMES, load_data

# define the methods
METHODS = {"mlp": "method_mlp", "transformer": "method_transformer", "pretrained": "method_pretrained_rl"}


def report(results, quiz, log=print):
    """
    The report function -- prints out a summary table.
    Used for the quiz

    Summary table (one row per result, 'before RL' rows included) + per-word table.
    """
    rows = []
    for r in results:
        rows += r.get("also", []) + [r]
    w = max(len(r["name"]) for r in rows)
    log("\n" + "=" * (w + 30))
    log(f"{'method':{w}s} | quiz accuracy")
    log("-" * (w + 30))
    for r in rows:
        log(f"{r['name']:{w}s} | {r['quiz_acc']:6.1%}  ({r['n_correct']}/{r['n']})")
    log("=" * (w + 30))
    log("Chance is 50%. With 40 words a 95% interval is about +/-15 points, so small gaps are not evidence.")
    if len(rows) > 1:
        log(f"\n{'word':14s}{'truth':9s}" + "".join(f"{i + 1:>5d}" for i in range(len(rows))) + "    (columns = rows above, in order)")
        for i, (word, label) in enumerate(quiz):
            cells = ["  ok " if r["quiz_pred"][i] == label else " WRONG" if r["quiz_pred"][i] >= 0 else "  n/a " for r in rows]
            log(f"{word:14s}{CLASS_NAMES[label]:9s}" + "".join(f"{c:>5s}" for c in cells))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Flags as helpers for different settings
    p.add_argument("method", choices=[*METHODS, "all"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--model", help="Hugging Face model id for method 3")
    p.add_argument("--steps", type=int, help="RL steps for method 3")
    p.add_argument("--batch", type=int, help="words per RL step")
    p.add_argument("--group", type=int, help="answers sampled per word")
    p.add_argument("--temperature", type=float, help="sampling temperature in RL (higher = more varied answers)")
    p.add_argument("--no-lora", dest="lora", action="store_const", const=False, help="fine-tune all weights")
    p.add_argument("--out", default="results_quiz")
    a = p.parse_args(argv)

    # load the dataset
    _, quiz, _ = load_data(strict=a.strict)
    # create an empty results list
    results = []
    # loop through all the selected methods
    for key in (METHODS if a.method == "all" else [a.method]):
        print(f"\n### {key}", flush=True)
        t0 = time.time()
        try:
            # dynamically import the model
            run = importlib.import_module(METHODS[key]).run
            # pass only the relevant arguments for the model
            kw = {k: v for k, v in vars(a).items() if v is not None and k in inspect.signature(run).parameters}
            # run the model and save the results
            results.append(run(log=lambda m: print(m, flush=True), **kw))
            # give timings too
            print(f"  done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep going so one missing dependency doesn't lose the other methods
            print(f"  {key} FAILED: {type(e).__name__}: {e}", flush=True)
            if a.method != "all":
                raise
    if not results:
        return
    rows = report(results, quiz)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    json.dump(dict(quiz=quiz, rows=rows), open(out / f"{a.method}.json", "w"), indent=1)
    try:
        from plot import plot_quiz
        plot_quiz([(r["name"], r["n_correct"]) for r in rows], out / f"{a.method}.png", n=len(quiz))
        print(f"\nsaved {out / (a.method + '.png')}")
    except ImportError as e:
        print(f"(chart skipped: {e})")


if __name__ == "__main__":
    main()
