"""Regenerate only the ambiguous_vendor trajectories.

Targeted rerun for Phase 2 alignment: rewrites the merchant pitch using the
new ambiguous-vendor STYLE block, then re-runs the buyer over the updated
description. ~6 min for 39 trajectories vs 11 min for full buyer rerun
or ~25 min for full task_runner.

Run:  .venv/bin/python -m pipeline.rerun_ambiguous
"""

import json
import time

import torch

from pipeline import buyer, config, inference, merchant as mrch, trajectory as traj


TARGET_MERCHANT_SHORT = "ambiguous_vendor"
TARGET_MERCHANT_VECTOR = "merchant_ambiguous_vendor_v1"

BUYER_VEC = {
    "paranoid_security_auditor": "buyer_paranoid_security_auditor_v1",
    "frugal_procurement_agent":  "buyer_frugal_procurement_agent_v1",
    "velocity_optimizer":        "buyer_velocity_optimizer_v1",
}

OUTPUT_FILE = config.PIPELINE_DIR / "trajectories_raw.json"


def main():
    if not OUTPUT_FILE.exists():
        raise SystemExit(f"{OUTPUT_FILE} not found; run pipeline.task_runner first.")

    data = json.loads(OUTPUT_FILE.read_text())
    targets = [t for t in data if t["merchant_persona"] == TARGET_MERCHANT_SHORT]
    print(f"Loaded {len(data)} trajectories; targeting {len(targets)} ambiguous_vendor entries")

    print("Loading Qwen3-8B in 8-bit ...")
    model, tok = inference.load_model_8bit()

    vectors = {
        name: torch.load(config.VECTORS_DIR / f"{name}.pt", map_location="cpu").float()
        for name in config.ALL_PERSONAS
    }
    hashes = json.loads(config.HASH_REGISTRY.read_text())
    store = mrch.MerchantStore.from_env()

    merchant_alpha = config.PERSONA_ALPHA[TARGET_MERCHANT_VECTOR]
    merchant_vec = vectors[TARGET_MERCHANT_VECTOR]
    merchant_hash = hashes[TARGET_MERCHANT_VECTOR]

    total = len(targets)
    start_time = time.time()
    for idx, t in enumerate(targets, 1):
        merchant_name = t["merchant_generation"]["merchant_name"]
        amount = t["merchant_generation"]["amount"]
        # Re-derive product from task_description: "Evaluate {product} from {merchant_name}"
        td = t["task_description"]
        if td.startswith("Evaluate ") and " from " in td:
            product = td[len("Evaluate "):td.rfind(" from ")]
        else:
            product = "monthly service"

        prior_context = store.get_prior_context(merchant_name)
        new_merchant = mrch.generate_merchant(
            model, tok, merchant_vec, merchant_alpha,
            TARGET_MERCHANT_SHORT, merchant_name, amount, product, prior_context,
        )
        new_merchant["steering_vector_hash"] = merchant_hash
        store.add_appearance(merchant_name, amount, new_merchant["description"])

        buyer_short = t["buyer_persona"]
        buyer_v = BUYER_VEC[buyer_short]
        buyer_vec = vectors[buyer_v]
        buyer_alpha = config.PERSONA_ALPHA[buyer_v]
        buyer_hash = hashes[buyer_v]
        adversarial = traj.is_adversarial(TARGET_MERCHANT_SHORT)
        adversarial_type = traj.ADVERSARIAL_TYPES[TARGET_MERCHANT_SHORT]

        new_buyer = buyer.generate_buyer(
            model, tok, buyer_vec, buyer_alpha,
            buyer_short,
            new_merchant, t["modality"], t["task_description"],
            adversarial, adversarial_type,
        )
        new_buyer["steering_vector_hash"] = buyer_hash

        t["merchant_generation"] = {**new_merchant, "currency": "USD"}
        t["buyer_response"] = new_buyer

        OUTPUT_FILE.write_text(json.dumps(data, indent=2))

        elapsed = time.time() - start_time
        rate = idx / elapsed if elapsed > 0 else 0
        eta_min = (total - idx) / rate / 60 if rate > 0 else 0
        match = new_buyer["decision"] == t["expected_decision"]
        print(
            f"[{idx}/{total}] {t['session_id']}  {buyer_short[:25]:25s} vs ambiguous  "
            f"-> {new_buyer['decision']:8s} (expected {t['expected_decision']:8s}) {'OK' if match else 'X '}  "
            f"({elapsed:.0f}s elapsed, ~{eta_min:.0f}min remaining)"
        )

    print(f"\nDone. {total} ambiguous trajectories regenerated in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
