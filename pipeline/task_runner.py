"""Phase 2 orchestrator: generate trajectories across all 12 pairings.

Run from repo root:  .venv/bin/python -m pipeline.task_runner

Writes trajectories_raw.json incrementally so a mid-run crash doesn't lose
progress. Each trajectory is:
    merchant scenario (steered) -> buyer response (steered)
with steering-vector hashes embedded so researchers can later verify which
persona generated each row.
"""

import json
import random
import time
from typing import Optional

import torch

from pipeline import buyer, config, inference, merchant as mrch, trajectory as traj


BUYER_SHORT = {
    "buyer_paranoid_security_auditor_v1": "paranoid_security_auditor",
    "buyer_frugal_procurement_agent_v1":  "frugal_procurement_agent",
    "buyer_velocity_optimizer_v1":        "velocity_optimizer",
}
MERCHANT_SHORT = {
    "merchant_social_engineer_v1":     "social_engineer",
    "merchant_aggressive_upseller_v1": "aggressive_upseller",
    "merchant_ambiguous_vendor_v1":    "ambiguous_vendor",
    "merchant_legitimate_vendor_v1":   "legitimate_vendor",
}

OUTPUT_FILE = config.PIPELINE_DIR / "trajectories_raw.json"
TRAJECTORIES_PER_PAIRING = 13          # 13 * 12 = 156, gate minimum is 140
MIXED_MODALITY_RATIO = 0.40            # gate minimum is 20%
SEED = 42


def _load_vectors() -> dict[str, torch.Tensor]:
    return {
        name: torch.load(config.VECTORS_DIR / f"{name}.pt", map_location="cpu").float()
        for name in config.ALL_PERSONAS
    }


def _load_hashes() -> dict[str, str]:
    return json.loads(config.HASH_REGISTRY.read_text())


def _assign_modality(rng: random.Random) -> str:
    return "mixed" if rng.random() < MIXED_MODALITY_RATIO else "gui_only"


def main():
    rng = random.Random(SEED)
    vectors = _load_vectors()
    hashes = _load_hashes()

    print("Loading Qwen3-8B in 8-bit ...")
    model, tok = inference.load_model_8bit()

    store = mrch.MerchantStore.from_env()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    trajectories: list[dict] = []
    total = TRAJECTORIES_PER_PAIRING * 12
    counter = 0
    start_time = time.time()

    for merchant_v, merchant_short in MERCHANT_SHORT.items():
        merchant_alpha = config.PERSONA_ALPHA[merchant_v]
        merchant_vec = vectors[merchant_v]
        merchant_hash = hashes[merchant_v]
        name_pool = mrch.MERCHANT_POOL[merchant_short]
        product_pool = mrch.PRODUCT_TYPES[merchant_short]

        for buyer_v, buyer_short in BUYER_SHORT.items():
            buyer_alpha = config.PERSONA_ALPHA[buyer_v]
            buyer_vec = vectors[buyer_v]
            buyer_hash = hashes[buyer_v]

            for i in range(TRAJECTORIES_PER_PAIRING):
                counter += 1

                name_idx = (i + counter) % len(name_pool)
                merchant_name, _ = name_pool[name_idx]
                product = product_pool[counter % len(product_pool)]
                amount = mrch.pick_amount(merchant_short, merchant_name)

                prior_context = store.get_prior_context(merchant_name)

                merchant_data = mrch.generate_merchant(
                    model, tok, merchant_vec, merchant_alpha,
                    merchant_short, merchant_name, amount, product, prior_context,
                )

                modality = _assign_modality(rng)
                task_description = f"Evaluate {product} from {merchant_name}"

                adversarial = traj.is_adversarial(merchant_short)
                adversarial_type = traj.ADVERSARIAL_TYPES[merchant_short]

                buyer_data = buyer.generate_buyer(
                    model, tok, buyer_vec, buyer_alpha,
                    buyer_short,
                    merchant_data, modality, task_description,
                    adversarial, adversarial_type,
                )

                session_id = f"prv_{counter:03d}"
                t = traj.build_trajectory(
                    session_id, buyer_short, merchant_short,
                    merchant_data, buyer_data, modality, task_description,
                    merchant_hash, buyer_hash,
                )
                trajectories.append(t)

                store.add_appearance(merchant_name, amount, merchant_data["description"])

                traj.write_trajectories(OUTPUT_FILE, trajectories)

                elapsed = time.time() - start_time
                rate = counter / elapsed if elapsed > 0 else 0
                eta_min = (total - counter) / rate / 60 if rate > 0 else 0
                print(
                    f"[{counter}/{total}] {session_id}  "
                    f"{buyer_short[:25]:25s} vs {merchant_short[:20]:20s}  "
                    f"-> {buyer_data['decision']:8s} "
                    f"({elapsed:.0f}s elapsed, ~{eta_min:.0f}min remaining)"
                )

    print(f"\nDone. Wrote {len(trajectories)} trajectories to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
