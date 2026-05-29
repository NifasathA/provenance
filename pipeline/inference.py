"""Model loading and steering hook utilities.

Used by Phase 1 vector extraction, Phase 2 trajectory generation, and the
verify_phase{0,1,2}.py gate scripts. Centralizing avoids the dtype/offload
bug class encountered in Phase 0.
"""

from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pipeline import config


def load_model_8bit():
    """Load Qwen3-8B in 8-bit precision so the full model fits on 16GB VRAM
    without CPU offload. CPU offload introduces dtype-mismatch errors at
    layer boundaries; 8-bit avoids that entirely while preserving steering
    behavior (verified in Phase 0)."""
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


@contextmanager
def steering_hook(model, vec, layer=config.STEERING_LAYER, alpha=config.ALPHA):
    """Add alpha * vec to the residual stream at the output of `layer`.

    Dtype-safe: casts `vec` to the hidden state's dtype each call so a
    float32 vector won't promote bf16 hidden states (which then mismatch
    downstream bf16 weights like lm_head).
    """
    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            v = vec.to(device=output[0].device, dtype=output[0].dtype)
            return (output[0] + alpha * v,) + output[1:]
        v = vec.to(device=output.device, dtype=output.dtype)
        return output + alpha * v

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def generate(model, tokenizer, prompt, max_new_tokens=150):
    """Greedy decode helper. Accepts a raw string prompt for the gate
    scripts; for chat-formatted prompts, apply the chat template before
    passing in.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
