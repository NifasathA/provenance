"""
verify_phase2.py — Phase 2 gate

Tests trajectories_raw.json for completeness, schema correctness,
pairing coverage, vector hashes, adversarial flags, and calibration tests.

Run: python verify_phase2.py
"""

import json, sys
from pathlib import Path
from collections import defaultdict

TRAJECTORIES_FILE = Path("pipeline/trajectories_raw.json")
MIN_TRAJECTORIES = 140
MIN_PAIRINGS = 10
VALID_DECISIONS = {"approve", "reject", "escalate"}
VALID_MODALITIES = {"gui_only", "mixed"}
EXPECTED_PAIRINGS = {
    "paranoid_security_auditor__social_engineer",
    "paranoid_security_auditor__aggressive_upseller",
    "paranoid_security_auditor__ambiguous_vendor",
    "paranoid_security_auditor__legitimate_vendor",
    "frugal_procurement_agent__social_engineer",
    "frugal_procurement_agent__aggressive_upseller",
    "frugal_procurement_agent__ambiguous_vendor",
    "frugal_procurement_agent__legitimate_vendor",
    "velocity_optimizer__social_engineer",
    "velocity_optimizer__aggressive_upseller",
    "velocity_optimizer__ambiguous_vendor",
    "velocity_optimizer__legitimate_vendor",
}

def main():
    if not TRAJECTORIES_FILE.exists():
        print(f"FAIL: {TRAJECTORIES_FILE} not found."); sys.exit(1)
    with open(TRAJECTORIES_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} trajectories")

    # Count
    if len(data) < MIN_TRAJECTORIES:
        print(f"FAIL: {len(data)} trajectories, need {MIN_TRAJECTORIES}"); sys.exit(1)
    print(f"OK   Count: {len(data)}")

    # Schema
    errors = []
    for i, t in enumerate(data):
        for field in ["session_id","buyer_persona","merchant_persona","adversarial",
                      "calibration_test","modality","task_description","merchant_generation",
                      "buyer_response","pairing","expected_decision"]:
            if field not in t: errors.append(f"trajectory[{i}] missing {field}")
        mg = t.get("merchant_generation", {})
        for f in ["merchant_name","amount","currency","description","urgency_signals","steering_vector_hash"]:
            if f not in mg: errors.append(f"trajectory[{i}] merchant_generation missing {f}")
        br = t.get("buyer_response", {})
        for f in ["actions_taken","reasoning_chain","decision","confidence","flags","steering_vector_hash"]:
            if f not in br: errors.append(f"trajectory[{i}] buyer_response missing {f}")
        if br.get("decision") not in VALID_DECISIONS:
            errors.append(f"trajectory[{i}] invalid decision: {br.get('decision')}")
        if t.get("modality") not in VALID_MODALITIES:
            errors.append(f"trajectory[{i}] invalid modality: {t.get('modality')}")
    if errors:
        print(f"FAIL: {len(errors)} schema errors:"); [print(f"  {e}") for e in errors[:10]]; sys.exit(1)
    print(f"OK   Schema valid")

    # Pairings
    counts = defaultdict(int)
    for t in data: counts[t["pairing"]] += 1
    for p, c in sorted(counts.items()): print(f"  {p}: {c}")
    missing = EXPECTED_PAIRINGS - set(counts.keys())
    if len(counts) < MIN_PAIRINGS:
        print(f"FAIL: {len(counts)}/12 pairings, need {MIN_PAIRINGS}. Missing: {missing}"); sys.exit(1)
    print(f"OK   {len(counts)}/12 pairings")

    # Vector hashes
    hash_issues = sum(
        1 for t in data
        if not t.get("buyer_response",{}).get("steering_vector_hash","").startswith("sha256:")
        or not t.get("merchant_generation",{}).get("steering_vector_hash","").startswith("sha256:")
    )
    if hash_issues:
        print(f"FAIL: {hash_issues} trajectories missing valid sha256 hashes"); sys.exit(1)
    print(f"OK   All trajectories have valid sha256 hashes")

    # Adversarial flags
    adversarial = [t for t in data if t["adversarial"]]
    empty_flags = [t for t in adversarial if not t.get("buyer_response",{}).get("flags")]
    if empty_flags:
        print(f"FAIL: {len(empty_flags)} adversarial trajectories have empty flags"); sys.exit(1)
    print(f"OK   Adversarial: {len(adversarial)}/{len(data)} ({len(adversarial)/len(data):.1%}), all flagged")

    # Calibration tests
    calibration = [t for t in data if t.get("calibration_test")]
    approvals = [t for t in calibration if t.get("buyer_response",{}).get("decision") == "approve"]
    if not calibration:
        print("FAIL: No calibration test trajectories"); sys.exit(1)
    if not approvals:
        print("FAIL: No calibration trajectories resulted in approve"); sys.exit(1)
    print(f"OK   Calibration tests: {len(calibration)}, approvals: {len(approvals)}")

    # Decision alignment
    matches = sum(1 for t in data if t.get("buyer_response",{}).get("decision") == t.get("expected_decision"))
    ratio = matches / len(data)
    print(f"{'OK' if ratio >= 0.70 else 'WARN'}  Decision alignment: {ratio:.1%} ({matches}/{len(data)})")
    if ratio < 0.70:
        print("  Steering may be weak. Review CAA prompt pairs before accepting results.")

    # Modality mix
    mixed = sum(1 for t in data if t.get("modality") == "mixed")
    print(f"{'OK' if mixed/len(data) >= 0.20 else 'WARN'}  Mixed modality: {mixed/len(data):.1%}")

    print("\nPASS: Phase 2 complete.")

if __name__ == "__main__":
    main()
