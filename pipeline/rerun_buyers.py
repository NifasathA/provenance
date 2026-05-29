"""Re-run only the buyer side over existing trajectories.

Keeps merchant_generation as-is (already persona-aligned) and regenerates
buyer_response using the current buyer prompt. Roughly 2x faster than the
full pipeline rerun.

Run:  .venv/bin/python -m pipeline.rerun_buyers
"""

import json
import time

import torch

from pipeline import buyer, config, inference, trajectory as traj


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
    print(f"Loaded {len(data)} trajectories")

    print("Loading Qwen3-8B in 8-bit ...")
    model, tok = inference.load_model_8bit()

    vectors = {
        name: torch.load(config.VECTORS_DIR / f"{name}.pt", map_location="cpu").float()
        for name in config.ALL_PERSONAS
    }

    total = len(data)
    start_time = time.time()
    for idx, t in enumerate(data, 1):
        buyer_short = t["buyer_persona"]
        merchant_short = t["merchant_persona"]
        vec_name = BUYER_VEC[buyer_short]
        vec = vectors[vec_name]
        alpha = config.PERSONA_ALPHA[vec_name]

        adversarial = traj.is_adversarial(merchant_short)
        adversarial_type = traj.ADVERSARIAL_TYPES[merchant_short]

        new_buyer = buyer.generate_buyer(
            model, tok, vec, alpha,
            buyer_short,
            t["merchant_generation"],
            t["modality"],
            t["task_description"],
            adversarial,
            adversarial_type,
        )
        # Preserve the original steering_vector_hash (it's still the same vector).
        new_buyer["steering_vector_hash"] = t["buyer_response"]["steering_vector_hash"]
        t["buyer_response"] = new_buyer

        # Incremental checkpoint.
        OUTPUT_FILE.write_text(json.dumps(data, indent=2))

        elapsed = time.time() - start_time
        rate = idx / elapsed if elapsed > 0 else 0
        eta_min = (total - idx) / rate / 60 if rate > 0 else 0
        match = new_buyer["decision"] == t["expected_decision"]
        print(
            f"[{idx}/{total}] {t['session_id']}  {buyer_short[:25]:25s} vs {merchant_short[:20]:20s}  "
            f"-> {new_buyer['decision']:8s} (expected {t['expected_decision']:8s}) {'OK' if match else 'X '}  "
            f"({elapsed:.0f}s elapsed, ~{eta_min:.0f}min remaining)"
        )

    print(f"\nDone. {total} buyer responses regenerated in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
