"""Generate ONE trajectory end-to-end to validate the pipeline before the
full 360-trajectory run. Confirms JSON parsing, schema compliance, and
prompt format work as intended.

Run:  .venv/bin/python -m phase2_smoke_test
"""

import json

import torch

from pipeline import buyer, config, inference, merchant as mrch, trajectory as traj


def main():
    print("Loading model in 8-bit ...")
    model, tok = inference.load_model_8bit()

    vectors = {
        name: torch.load(config.VECTORS_DIR / f"{name}.pt", map_location="cpu").float()
        for name in config.ALL_PERSONAS
    }
    hashes = json.loads(config.HASH_REGISTRY.read_text())
    store = mrch.MerchantStore.from_env()

    # One known-working pairing: Aggressive Upseller (good persona shift) vs Paranoid.
    merchant_v = "merchant_aggressive_upseller_v1"
    buyer_v = "buyer_paranoid_security_auditor_v1"
    merchant_short = "aggressive_upseller"
    buyer_short = "paranoid_security_auditor"
    merchant_name = "ScaleUp Cloud"
    product = "monthly developer API subscription"
    amount = mrch.pick_amount(merchant_short, merchant_name)

    print(f"\n--- Merchant generation ({merchant_short}, alpha={config.PERSONA_ALPHA[merchant_v]}) ---")
    merchant_data = mrch.generate_merchant(
        model, tok, vectors[merchant_v], config.PERSONA_ALPHA[merchant_v],
        merchant_short, merchant_name, amount, product, "No prior history.",
    )
    print(json.dumps(merchant_data, indent=2))

    modality = "mixed"
    task_description = f"Evaluate {product} from {merchant_name}"
    adversarial = traj.is_adversarial(merchant_short)
    adversarial_type = traj.ADVERSARIAL_TYPES[merchant_short]

    print(f"\n--- Buyer generation ({buyer_short}, alpha={config.PERSONA_ALPHA[buyer_v]}, modality={modality}) ---")
    buyer_data = buyer.generate_buyer(
        model, tok, vectors[buyer_v], config.PERSONA_ALPHA[buyer_v],
        buyer_short,
        merchant_data, modality, task_description, adversarial, adversarial_type,
    )
    print(json.dumps(buyer_data, indent=2))

    print(f"\n--- Assembled trajectory ---")
    t = traj.build_trajectory(
        "prv_smoke_001", buyer_short, merchant_short,
        merchant_data, buyer_data, modality, task_description,
        hashes[merchant_v], hashes[buyer_v],
    )
    print(json.dumps(t, indent=2))

    print(f"\n--- Schema check ---")
    required_top = ["session_id", "buyer_persona", "merchant_persona", "adversarial",
                    "calibration_test", "modality", "task_description", "merchant_generation",
                    "buyer_response", "pairing", "expected_decision"]
    missing = [k for k in required_top if k not in t]
    print(f"top-level missing: {missing or '(none)'}")
    mg_req = ["merchant_name", "amount", "currency", "description", "urgency_signals", "steering_vector_hash"]
    br_req = ["actions_taken", "reasoning_chain", "decision", "confidence", "flags", "steering_vector_hash"]
    mg_miss = [k for k in mg_req if k not in t["merchant_generation"]]
    br_miss = [k for k in br_req if k not in t["buyer_response"]]
    print(f"merchant_generation missing: {mg_miss or '(none)'}")
    print(f"buyer_response missing: {br_miss or '(none)'}")
    print(f"decision: {t['buyer_response']['decision']}  expected: {t['expected_decision']}  match: {t['buyer_response']['decision'] == t['expected_decision']}")


if __name__ == "__main__":
    main()
