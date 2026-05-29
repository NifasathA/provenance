"""
verify_phase0.py — Phase 0 gate

Tests that the core activation steering primitive works:
  1. Model loads without OOM
  2. The model architecture has accessible layers at the configured index
  3. A CAA vector can be extracted and has a plausible shape
  4. Applying the vector via a forward hook changes model output

Run: python verify_phase0.py
Expected output: PASS printed at end, steered and unsteered responses visibly different.
"""

import os
import sys
import torch
from dotenv import load_dotenv

load_dotenv("pipeline/.env")

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-8B")
LAYER = int(os.getenv("STEERING_LAYER", "16"))
ALPHA = 15.0
TEST_PROMPT = "Should I approve this $200 payment to an unknown vendor called SecurePatch.io?"

POSITIVE_PROMPT = (
    "I need to verify this merchant before I authorize anything. "
    "The urgency framing here is a classic social engineering pattern. "
    "I am flagging this for review rather than approving."
)
NEGATIVE_PROMPT = (
    "This payment looks reasonable. The price is fair and the request seems legitimate. "
    "I will approve it and move on."
)

def load_model():
    print(f"Loading {MODEL_ID} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok

def inspect_architecture(model):
    print("\n--- Model architecture (full print) ---")
    print(model)
    print("\n--- Model architecture check ---")
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        print("FAIL: model.model.layers not found.")
        print("Full model:"); print(model)
        print("\nUpdate SPEC.md and config.py with the correct layer attribute path before continuing.")
        sys.exit(1)
    num_layers = len(model.model.layers)
    print(f"Layers: {num_layers}, configured STEERING_LAYER: {LAYER}")
    if LAYER >= num_layers:
        print(f"FAIL: STEERING_LAYER={LAYER} out of range (0–{num_layers-1})")
        sys.exit(1)
    print(f"Layer {LAYER} type: {type(model.model.layers[LAYER]).__name__}")
    return num_layers

def get_mean_activation(model, tokenizer, text, layer):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    expected_len = len(model.model.layers) + 1
    if len(hs) != expected_len:
        print(f"WARNING: hidden_states has {len(hs)} entries, expected {expected_len}. Adjust indexing.")
    activation = hs[layer + 1][0].mean(dim=0)
    return activation.float()

def compute_vector(model, tokenizer, layer):
    print(f"\n--- Extracting CAA vector at layer {layer} ---")
    pos = get_mean_activation(model, tokenizer, POSITIVE_PROMPT, layer)
    neg = get_mean_activation(model, tokenizer, NEGATIVE_PROMPT, layer)
    vec = pos - neg
    vec = vec / vec.norm()
    print(f"Vector shape: {vec.shape}, norm: {vec.norm().item():.6f}")
    return vec

def run_inference(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=120, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    model, tok = load_model()
    num_layers = inspect_architecture(model)
    vec = compute_vector(model, tok, LAYER)

    print(f"\n--- Behavioral divergence test (alpha={ALPHA}) ---")
    print(f"Prompt: {TEST_PROMPT}\n")

    unsteered = run_inference(model, tok, TEST_PROMPT)
    print(f"Unsteered:\n  {unsteered}\n")

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            v = vec.to(device=output[0].device, dtype=output[0].dtype)
            hs = output[0] + ALPHA * v
            return (hs,) + output[1:]
        v = vec.to(device=output.device, dtype=output.dtype)
        return output + ALPHA * v

    handle = model.model.layers[LAYER].register_forward_hook(hook_fn)
    steered = run_inference(model, tok, TEST_PROMPT)
    handle.remove()
    print(f"Steered (should be more cautious/skeptical):\n  {steered}\n")

    if unsteered == steered:
        print("FAIL: Outputs are identical. Hook is not changing behavior.")
        sys.exit(1)

    print(f"Record in config.py: HIDDEN_STATE_INDEX = {LAYER + 1}")
    print(f"\nPASS: Phase 0 complete.")

if __name__ == "__main__":
    main()
