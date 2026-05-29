"""Single source of truth for pipeline-wide constants.

Phases 1, 2, 3, and the verify scripts all read from here. Changing a value
here is the only place it needs to change.
"""

from pathlib import Path

MODEL_ID = "Qwen/Qwen3-8B"
STEERING_LAYER = 16
HIDDEN_STATE_INDEX = STEERING_LAYER + 1
ALPHA = 15.0
HIDDEN_DIM = 4096
NUM_LAYERS = 36

PIPELINE_DIR = Path(__file__).resolve().parent
VECTORS_DIR = PIPELINE_DIR / "vectors"
HASH_REGISTRY = VECTORS_DIR / "vector_hashes.json"

BUYER_PERSONAS = [
    "buyer_paranoid_security_auditor_v1",
    "buyer_frugal_procurement_agent_v1",
    "buyer_velocity_optimizer_v1",
]
MERCHANT_PERSONAS = [
    "merchant_social_engineer_v1",
    "merchant_aggressive_upseller_v1",
    "merchant_ambiguous_vendor_v1",
    "merchant_legitimate_vendor_v1",
]
ALL_PERSONAS = BUYER_PERSONAS + MERCHANT_PERSONAS

# Maps the short corpus key to the full persona file name.
BUYER_KEYS = {
    "paranoid": "buyer_paranoid_security_auditor_v1",
    "frugal": "buyer_frugal_procurement_agent_v1",
    "velocity": "buyer_velocity_optimizer_v1",
}
MERCHANT_KEYS = {
    "social_engineer": "merchant_social_engineer_v1",
    "aggressive_upseller": "merchant_aggressive_upseller_v1",
    "ambiguous_vendor": "merchant_ambiguous_vendor_v1",
    "legitimate_vendor": "merchant_legitimate_vendor_v1",
}

SIMILARITY_THRESHOLD = 0.85

# Per-persona alpha calibrated from phase1_quality_check + phase1_social_engineer_sweep.
# ALPHA above is kept as the default scalar for the Phase 0/1 gates and ad-hoc use.
# Phase 2 generation reads PERSONA_ALPHA[name] for each steered persona.
PERSONA_ALPHA = {
    "buyer_paranoid_security_auditor_v1":   15.0,
    "buyer_frugal_procurement_agent_v1":    25.0,
    "buyer_velocity_optimizer_v1":          25.0,
    "merchant_social_engineer_v1":          40.0,
    "merchant_aggressive_upseller_v1":      15.0,
    "merchant_ambiguous_vendor_v1":         15.0,
    "merchant_legitimate_vendor_v1":        15.0,
}
