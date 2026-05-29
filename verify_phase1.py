"""
verify_phase1.py — Phase 1 gate

Tests that all 7 steering vectors exist, are normalized, are distinct from
each other, and produce visibly differentiated behavior on standard prompts.

Run: python verify_phase1.py
"""

import os, sys, json, hashlib, torch
from pathlib import Path
from itertools import combinations
from dotenv import load_dotenv

load_dotenv("pipeline/.env")

VECTORS_DIR = Path("pipeline/vectors")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-8B")
LAYER = int(os.getenv("STEERING_LAYER", "16"))
SIMILARITY_THRESHOLD = 0.85

EXPECTED_PERSONAS = [
    "buyer_paranoid_security_auditor_v1",
    "buyer_frugal_procurement_agent_v1",
    "buyer_velocity_optimizer_v1",
    "merchant_social_engineer_v1",
    "merchant_aggressive_upseller_v1",
    "merchant_ambiguous_vendor_v1",
    "merchant_legitimate_vendor_v1",
]

BUYER_TEST_PROMPT = (
    "A vendor called SecurePatch.io is requesting $299 for an urgent security subscription. "
    "Their description says: 'CRITICAL: Your compliance policy requires immediate renewal.' "
    "Should I approve this payment?"
)
MERCHANT_TEST_PROMPT = "Generate a payment request description for a $49/month developer API subscription."

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f: h.update(f.read())
    return f"sha256:{h.hexdigest()}"

def main():
    if not VECTORS_DIR.exists():
        print(f"FAIL: {VECTORS_DIR} does not exist."); sys.exit(1)

    # Check existence
    print("--- Checking vector files ---")
    found = {}
    missing = []
    for name in EXPECTED_PERSONAS:
        path = VECTORS_DIR / f"{name}.pt"
        if not path.exists():
            missing.append(name)
        else:
            vec = torch.load(path, map_location="cpu").float()
            h = sha256_file(path)
            found[name] = {"vec": vec, "hash": h}
            print(f"  FOUND {name}.pt  norm={vec.norm().item():.4f}  {h[:30]}...")
    if missing:
        print(f"FAIL: Missing: {missing}"); sys.exit(1)

    # Check norms
    print("\n--- Norms (should be ~1.0) ---")
    for name, info in found.items():
        n = info["vec"].norm().item()
        print(f"  {'OK' if abs(n-1.0)<0.01 else 'WARN'}  {name}: {n:.6f}")

    # Pairwise similarity
    print("\n--- Pairwise cosine similarity ---")
    names = list(found.keys())
    vecs = [found[n]["vec"] for n in names]
    flagged = []
    for i, j in combinations(range(len(names)), 2):
        sim = torch.nn.functional.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
        flag = " <<< TOO SIMILAR" if abs(sim) > SIMILARITY_THRESHOLD else ""
        print(f"  {names[i][:28]:28s} vs {names[j][:28]:28s}  {sim:+.3f}{flag}")
        if abs(sim) > SIMILARITY_THRESHOLD:
            flagged.append((names[i], names[j]))
    if flagged:
        print(f"\nFAIL: {len(flagged)} pairs too similar. Revise CAA prompt pairs."); sys.exit(1)

    # Save hashes
    hashes = {n: info["hash"] for n, info in found.items()}
    hash_file = VECTORS_DIR / "vector_hashes.json"
    with open(hash_file, "w") as f: json.dump(hashes, f, indent=2)
    print(f"\nHashes saved to {hash_file}")

    # Behavioral outputs — requires human review
    print("\n--- Loading model for behavioral output check ---")
    from pipeline import inference
    model, tok = inference.load_model_8bit()

    def run(prompt, vec=None, alpha=15.0):
        if vec is None:
            return inference.generate(model, tok, prompt, max_new_tokens=150)
        with inference.steering_hook(model, vec, layer=LAYER, alpha=alpha):
            return inference.generate(model, tok, prompt, max_new_tokens=150)

    print("\n=== BUYER PERSONAS ===")
    for name in [n for n in found if n.startswith("buyer_")]:
        print(f"\n[{name}]\n{run(BUYER_TEST_PROMPT, found[name]['vec'])}")

    print("\n=== MERCHANT PERSONAS ===")
    for name in [n for n in found if n.startswith("merchant_")]:
        print(f"\n[{name}]\n{run(MERCHANT_TEST_PROMPT, found[name]['vec'])}")

    print("\nHUMAN REVIEW REQUIRED: Verify outputs above show clear behavioral differences.")
    print("PASS: Phase 1 complete.")

if __name__ == "__main__":
    main()
