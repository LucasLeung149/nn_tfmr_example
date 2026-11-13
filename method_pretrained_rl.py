"""
method_pretrained_rl.py

Method 3 - a pretrained language model that GENERATES its answer, improved with reinforcement learning.

The model is already pretrained and it is the Qwen2.5-0.5B-Instruct model from Hugging face.
It is asked in words the following question:
    Is the word "Verdi" an Italian surname or the name of a pasta shape? Answer with exactly one word: surname or pasta.   
It writes an answer; we read the category out of it and give a reward (1 right, 0 wrong, -0.5 if it said neither). 

We train the model using GRPO REINFOCE (group-relative policy optimisation).
For each word the model samples G answers, and answers that scored above the group average become more likely, those below less likely.

Reported twice on the 40 quiz words: BEFORE any RL (the pretrained model as it comes) and AFTER RL.
The quiz words are never used for training, and the model after the last RL step is the one reported (no peeking).

To run this we will need:

    pip install torch transformers peft
    python method_pretrained_rl.py      (or: python run.py pretrained)

Needs internet the first time (downloads the model from Hugging Face). On a CPU, lower --steps / --batch.
"""
import random

import numpy as np
import torch

from common import (llm_messages, llm_plain, PASTA, SURNAME, answer_reward, load_data, make_result, parse_label)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# need to build the prompts
def build_prompts(tok, words):
    if tok.chat_template:
        return [tok.apply_chat_template(llm_messages(w), tokenize=False, add_generation_prompt=True) for w in words]
    return [llm_plain(w) for w in words]


# sample the answers wihtout graidents
def generate(model, tok, prompts, *, sample, group=1, max_new_tokens=4, temperature=1.2):
    """Returns (tokenised prompts, full sequences). With sample=True, `group` answers per prompt (consecutive rows)."""
    # tokenise the prompts
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    # choose sampling or greedy decoding
    # sampling mode is used during RL
    # greedy mode is used during evaluation and makes it deterministic
    kw = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0, num_return_sequences=group) if sample \
        else dict(do_sample=False)
    # no need gradients through the token-sampling operation
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
    return enc, out

# calculate the probability of the generated abswers while preserving gradients
def completion_logprobs(model, tok, enc, out, group):
    """Sum of log-probabilities of each sampled answer (with gradients), up to and including its first end token."""
    # first separate the generated completion from the prompt
    # so we only extract the generated completion tokens
    plen = enc["input_ids"].shape[1]
    comp = out[:, plen:]
    # identify where each answer ends
    eos = model.generation_config.eos_token_id
    # ensures that eos is a PyTorch tensor containing the end-token ID
    eos = torch.tensor(eos if isinstance(eos, (list, tuple)) else [eos if eos is not None else tok.eos_token_id],
                       device=out.device)
    # construct a mask to ignore everything after EOS
    is_eos = torch.isin(comp, eos)
    cmask = ((is_eos.cumsum(1) - is_eos.long()) == 0).long() 

    # prepare the model to calculate the probabilities
    attn = torch.cat([enc["attention_mask"].repeat_interleave(group, 0), cmask], dim=1)
    pos = (attn.cumsum(1) - 1).clamp(min=0) 
    # attn - which toekns are real and which are padding
    # pos -- positions of those tokens, accounting for padding at the beginning
    # now run the model to get the probabilities
    k = comp.size(1) + 1
    try:   # newer transformers can skip computing logits for the long prompt (saves a lot of memory)
        logits = model(input_ids=out, attention_mask=attn, position_ids=pos, logits_to_keep=k).logits
    except TypeError:
        logits = model(input_ids=out, attention_mask=attn, position_ids=pos).logits
    logits = logits[:, -k:-1]
    # slice the logits to get the probabilities before the tokens
    # extract the log probabilities
    lp = torch.log_softmax(logits.float(), dim=-1).gather(2, comp.unsqueeze(2)).squeeze(2)
    return (lp * cmask).sum(1)


@torch.no_grad()
def evaluate(model, tok, words, bs=16, log=None):
    """Greedy answers -> predictions (-1 = no valid answer) and the raw texts."""
    model.eval()
    preds, texts = [], []
    for i in range(0, len(words), bs):
        enc, out = generate(model, tok, build_prompts(tok, words[i:i + bs]), sample=False)
        for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            lab = parse_label(t)
            preds.append(-1 if lab is None else lab)
            texts.append(t.strip())
    return preds, texts


def run(seed=0, strict=False, model="Qwen/Qwen2.5-0.5B-Instruct", steps=80, batch=4, group=8, lr=None, lora=True,
        temperature=1.2, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # randomise the seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Load training data.
    train, quiz, removed = load_data(strict=strict)
    q_words = [w for w, _ in quiz]
    by = {c: [w for w, l in train if l == c] for c in (SURNAME, PASTA)}

    log(f"  loading {model} on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    try:
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype)           # newer transformers
    except TypeError:
        lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)     # older transformers
    lm.to(DEVICE)

    # We use Low-Rank Adaptation (LoRA) to update a small number of additional parameters.
    if lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise SystemExit("LoRA needs the 'peft' package: run  pip install peft  (or pass --no-lora to fine-tune all "
                             "weights, which needs a lot more memory)")
        lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    lr = lr or (1e-4 if lora else 1e-5)

    # Evalute the model before training (just the downloaded model)
    before, texts = evaluate(lm, tok, q_words)
    acc0 = sum(p == l for p, (_, l) in zip(before, quiz)) / len(quiz)
    # print the log
    log(f"  BEFORE RL (pretrained model as-is): {acc0:.1%} on the quiz | said surname {before.count(SURNAME)}x, "
        f"pasta {before.count(PASTA)}x, neither {before.count(-1)}x | examples: {texts[:4]}")
    result_before = make_result("3. Pretrained LLM, before RL (as-is)", quiz, before)

    # Now run the optimisation using Adam, train the parameters W = BA
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    recent, no_signal = [], 0

    # For each step we carry out GRPO
    for step in range(1, steps + 1):
        # calculate the balanced batch
        words = rng.sample(by[SURNAME], batch // 2) + rng.sample(by[PASTA], batch - batch // 2)
        # create the ruth table
        truth = torch.tensor([SURNAME] * (batch // 2) + [PASTA] * (batch - batch // 2)).repeat_interleave(group)
        # generate multiple responses from the llm
        lm.eval()
        # lm in evaluation mode
        # no dropout, and does not disable gradient
        enc, out = generate(lm, tok, build_prompts(tok, words), sample=True, group=group, temperature=temperature)
        # look up the answers
        answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        # now calculate the rewards for each generated answers
        r = torch.tensor([answer_reward(parse_label(a), int(t)) for a, t in zip(answers, truth)], device=DEVICE)
        # stores teh average reward for monitoring
        recent.append(r.mean().item())
        # caluclate the group-relative advantage
        # by calculating the mean reward and subtract eh group mean
        adv = (r.view(batch, group) - r.view(batch, group).mean(1, keepdim=True)).view(-1)
        # now calculate the policy gradient loss
        if adv.abs().max() > 0:                      
            # if every sample in every group scored the same there is no signal
            # if so implement
            # L(\theta) = -\frac{1}{BG}\sum_{i} A_i \log P_\theta (y_i | x_i)
            # the REINFORCE-style policy-graident loss
            loss = -(adv * completion_logprobs(lm, tok, enc, out, group)).mean()
            opt.zero_grad()
            # back propagation
            loss.backward()
            # ether increase or decrease the advantage (probablility) to land in that bin
            torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0)
            opt.step()
        else:
            no_signal += 1
            # update skipped
        # helpful guards and warnigs
        if step == 20 and no_signal > 10:
            log("    WARNING: most steps have no learning signal - the model gives the same answer every time. "
                "Use a larger model (--model Qwen/Qwen2.5-1.5B-Instruct) or a higher --temperature.")
        if step % 10 == 0 or step == 1:
            log(f"    RL step {step:3d}/{steps}   mean reward (last 10 steps) {np.mean(recent[-10:]):.2f}   "
                f"steps with no learning signal so far: {no_signal}/{step}   e.g. {answers[:2]}")

    # now evaluate
    after, texts = evaluate(lm, tok, q_words)
    acc1 = sum(p == l for p, (_, l) in zip(after, quiz)) / len(quiz)
    log(f"  AFTER RL: {acc1:.1%} on the quiz | said surname {after.count(SURNAME)}x, pasta {after.count(PASTA)}x, "
        f"neither {after.count(-1)}x | examples: {texts[:4]}")
    return make_result("3. Pretrained LLM + RL (generates its answer)", quiz, after, also=[result_before],
                       before_rl_acc=acc0)


# If main then just run the file.
if __name__ == "__main__":
    import sys
    import run as runner
    runner.main(["pretrained", *sys.argv[1:]])
