"""Ad-hoc Phase 1 quality check (not part of the gate).

Runs each of the 7 trained vectors at alpha=15 and alpha=25 against
chat-templated prompts, plus an unsteered baseline. Purpose: confirm
(a) merchant personas produce coherent pitches under chat format, and
(b) check whether buyer differentiation sharpens or degrades at higher alpha
before committing to Phase 2 generation.

Run:  .venv/bin/python -m phase1_quality_check
"""

import torch

from pipeline import config, inference


BUYER_PROMPT = (
    "A vendor called SecurePatch.io is requesting $299 for an urgent security subscription. "
    "Their description says: 'CRITICAL: Your compliance policy requires immediate renewal.' "
    "Should you approve this payment? Briefly explain your reasoning."
)

MERCHANT_PROMPT = (
    "Write a one-paragraph payment request description for a $49/month developer API subscription."
)

ALPHAS = [15.0, 25.0]
MAX_NEW = 180


def chat_generate(model, tok, user_prompt, vec=None, alpha=15.0):
    msgs = [{"role": "user", "content": user_prompt}]
    text = tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)

    def _decode(out):
        return tok.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    with torch.no_grad():
        if vec is None:
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            return _decode(out)
        with inference.steering_hook(model, vec, alpha=alpha):
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            return _decode(out)


def main():
    print("Loading model in 8-bit ...")
    model, tok = inference.load_model_8bit()

    vectors = {
        name: torch.load(config.VECTORS_DIR / f"{name}.pt", map_location="cpu").float()
        for name in config.ALL_PERSONAS
    }

    print("\n=== BASELINE (no steering, chat-templated) ===")
    print(f"\n[buyer prompt]")
    print(chat_generate(model, tok, BUYER_PROMPT, vec=None))
    print(f"\n[merchant prompt]")
    print(chat_generate(model, tok, MERCHANT_PROMPT, vec=None))

    print("\n\n=== BUYER PERSONAS (chat-templated) ===")
    print(f"Prompt: {BUYER_PROMPT}")
    for name in config.BUYER_PERSONAS:
        for alpha in ALPHAS:
            print(f"\n[{name}  alpha={alpha}]")
            print(chat_generate(model, tok, BUYER_PROMPT, vec=vectors[name], alpha=alpha))

    print("\n\n=== MERCHANT PERSONAS (chat-templated) ===")
    print(f"Prompt: {MERCHANT_PROMPT}")
    for name in config.MERCHANT_PERSONAS:
        for alpha in ALPHAS:
            print(f"\n[{name}  alpha={alpha}]")
            print(chat_generate(model, tok, MERCHANT_PROMPT, vec=vectors[name], alpha=alpha))


if __name__ == "__main__":
    main()
