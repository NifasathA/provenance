"""Build 7 persona steering vectors from the contrast corpus.

Run from repo root:  .venv/bin/python -m pipeline.train_vectors

For each scenario we run one forward pass per persona response and cache the
last-token activation. Then for each persona P we form a CAA vector from
mean(P's activations) - mean(other-persona activations on the same scenarios).
Vectors are unit-normalized and saved as pipeline/vectors/{name}.pt; SHA-256
hashes are written to pipeline/vectors/vector_hashes.json.
"""

import hashlib
import json
from itertools import combinations

import torch

from pipeline import caa, config, corpus, inference


def _chat(user, assistant):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _compute_buyer_activations(model, tok):
    """Cache layer-output activations for every (scenario_idx, persona_key)."""
    activations = []
    for s in corpus.BUYER_SCENARIOS:
        per_key = {}
        for key in config.BUYER_KEYS:
            msgs = _chat(s["scenario"], s[key])
            per_key[key] = caa.get_last_token_activation(
                model, tok, msgs, config.STEERING_LAYER
            )
        activations.append(per_key)
    return activations


def _compute_merchant_activations(model, tok):
    """Cache activations for every (product_idx, merchant_key)."""
    activations = []
    for s in corpus.MERCHANT_SCENARIOS:
        per_key = {}
        user_prompt = f"Write a payment request description for this product: {s['product']}"
        for key in config.MERCHANT_KEYS:
            msgs = _chat(user_prompt, s[key])
            per_key[key] = caa.get_last_token_activation(
                model, tok, msgs, config.STEERING_LAYER
            )
        activations.append(per_key)
    return activations


def _build_vectors(cached_activations, key_to_name):
    """For each target persona, positives = its activations across scenarios,
    negatives = the other personas' activations across the same scenarios."""
    keys = list(key_to_name.keys())
    vectors = {}
    for target in keys:
        pos = [scn[target] for scn in cached_activations]
        neg = [scn[other] for scn in cached_activations for other in keys if other != target]
        vectors[key_to_name[target]] = caa.caa_vector(pos, neg)
    return vectors


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return f"sha256:{h.hexdigest()}"


def _report_pairwise(vectors):
    names = list(vectors.keys())
    print("\n--- Pairwise cosine similarity ---")
    flagged = []
    for i, j in combinations(range(len(names)), 2):
        sim = torch.nn.functional.cosine_similarity(
            vectors[names[i]].unsqueeze(0),
            vectors[names[j]].unsqueeze(0),
        ).item()
        flag = " <<< TOO SIMILAR" if abs(sim) > config.SIMILARITY_THRESHOLD else ""
        print(f"  {names[i][:32]:32s} vs {names[j][:32]:32s}  {sim:+.3f}{flag}")
        if abs(sim) > config.SIMILARITY_THRESHOLD:
            flagged.append((names[i], names[j], sim))
    return flagged


def main():
    config.VECTORS_DIR.mkdir(exist_ok=True)

    print("Loading Qwen3-8B in 8-bit ...")
    model, tok = inference.load_model_8bit()

    print(f"\n--- Caching activations at layer {config.STEERING_LAYER} ---")
    print(f"Buyer scenarios: {len(corpus.BUYER_SCENARIOS)} x 3 personas = {len(corpus.BUYER_SCENARIOS)*3} forwards")
    buyer_acts = _compute_buyer_activations(model, tok)
    print(f"Merchant scenarios: {len(corpus.MERCHANT_SCENARIOS)} x 4 personas = {len(corpus.MERCHANT_SCENARIOS)*4} forwards")
    merchant_acts = _compute_merchant_activations(model, tok)

    print("\n--- Building CAA vectors ---")
    vectors = {}
    vectors.update(_build_vectors(buyer_acts, config.BUYER_KEYS))
    vectors.update(_build_vectors(merchant_acts, config.MERCHANT_KEYS))

    print("\n--- Saving vectors ---")
    for name, vec in vectors.items():
        path = config.VECTORS_DIR / f"{name}.pt"
        torch.save(vec, path)
        print(f"  {path.name}  shape={tuple(vec.shape)}  norm={vec.norm().item():.6f}")

    hashes = {name: _sha256(config.VECTORS_DIR / f"{name}.pt") for name in vectors}
    config.HASH_REGISTRY.write_text(json.dumps(hashes, indent=2))
    print(f"\nHashes written to {config.HASH_REGISTRY}")
    for name, h in hashes.items():
        print(f"  {name}: {h[:30]}...")

    flagged = _report_pairwise(vectors)
    if flagged:
        print(f"\n{len(flagged)} pair(s) above threshold ({config.SIMILARITY_THRESHOLD}). "
              "Re-run train_vectors after revising corpus, or apply caa.orthogonalize.")
    else:
        print(f"\nAll pairs below {config.SIMILARITY_THRESHOLD} similarity threshold.")


if __name__ == "__main__":
    main()
