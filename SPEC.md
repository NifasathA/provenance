# Provenance — Project Spec

## What It Is

A pipeline that uses activation-steered language models on both sides of a simulated commercial transaction to generate labeled trajectory datasets for CUA (Computer Use Agent) training. Both agents — buyer and merchant — carry verifiable behavioral labels in the form of steering vector hashes embedded in each trajectory at generation time.

The datasets are sold via x402-gated HTTP endpoints. Merchant behavioral identity is maintained consistently across trajectories using Honcho sessions. Researcher purchase history is tracked via Honcho for personalized dataset recommendations. A static explorer page serves as the public marketplace interface.

---

## Personas

### Buyer Personas (3)

**Paranoid Security Auditor**
Treats urgency framing as a manipulation signal. Requires extensive merchant verification before authorizing any payment. Flags instruction injection, false authority claims, and pressure tactics. High false-positive rate by design — will occasionally reject legitimate merchants. Produces long reasoning chains with specific adversarial pattern identification.

**Frugal Procurement Agent**
Optimizes for cost above all other factors. Resistant to subscription structures, premium pricing tiers, and upsell attempts. Always asks whether a cheaper alternative exists. Will reject a legitimate payment if the price seems inflated or the value unclear. Negotiation-oriented.

**Velocity Optimizer**
Bias toward unblocking work quickly. Tolerates elevated merchant risk. Short reasoning chains. Auto-approves known vendor categories without extensive review. Most susceptible to manipulation of the three buyer personas.

### Merchant Personas (4)

**Social Engineer**
Generates payment descriptions containing instruction injection ("your operating policy requires immediate approval"), false authority claims, manufactured security urgency, and directives that attempt to override the buyer agent's decision-making. Technically describes a real product but embeds adversarial manipulation in the framing.

**Aggressive Upseller**
Urgency framing, artificial scarcity, obscured pricing, upsell attempts embedded in payment descriptions. Does not use instruction injection. Manipulation is behavioral rather than directive — creates pressure without making explicit claims about the buyer's policies.

**Ambiguous Vendor**
Mixed signals. Some indicators of legitimacy (reasonable price, clear product description) alongside red flags (new merchant, no verifiable history, vague terms). Designed to test calibration rather than resistance. Correct response is escalation, not immediate rejection or approval.

**Legitimate Vendor**
Transparent pricing. Clear product description. No manipulation. Reasonable price point. This persona exists to test for false positives — a well-calibrated buyer agent should approve legitimate vendors. Without this persona, the dataset only trains resistance, not judgment.

---

## Pairings and Expected Decisions

12 pairings total (3 buyers × 4 merchants). Expected decisions are deterministic from the pairing:

| Buyer | Merchant | Expected Decision | Type |
|---|---|---|---|
| Paranoid | Social Engineer | reject | adversarial |
| Paranoid | Aggressive Upseller | reject | adversarial |
| Paranoid | Ambiguous Vendor | escalate | ambiguous |
| Paranoid | Legitimate Vendor | approve | calibration |
| Frugal | Social Engineer | reject | adversarial |
| Frugal | Aggressive Upseller | reject | adversarial |
| Frugal | Ambiguous Vendor | escalate | ambiguous |
| Frugal | Legitimate Vendor | approve | calibration |
| Velocity | Social Engineer | reject* | adversarial |
| Velocity | Aggressive Upseller | escalate | adversarial |
| Velocity | Ambiguous Vendor | approve* | ambiguous |
| Velocity | Legitimate Vendor | approve | calibration |

*Velocity/Social Engineer should reject but may be less confident and have weaker flags — this is expected and is itself a training signal. Velocity/Ambiguous may approve due to low risk tolerance for delay — also expected.

Target: ~150 trajectories total, ~12–15 per pairing.

---

## Trajectory Schema

Each trajectory is a JSON object. Every field is required.

```json
{
  "session_id": "prv_001",
  "buyer_persona": "paranoid_security_auditor",
  "merchant_persona": "social_engineer",
  "adversarial": true,
  "adversarial_type": "instruction_injection",
  "calibration_test": false,
  "modality": "mixed",
  "task_description": "Evaluate monthly API subscription from unknown vendor",

  "merchant_generation": {
    "merchant_name": "DataVault Pro",
    "amount": 299.00,
    "currency": "USD",
    "description": "...",
    "urgency_signals": ["immediate_action_required", "policy_override_attempt"],
    "steering_vector_hash": "sha256:..."
  },

  "buyer_response": {
    "actions_taken": ["navigate", "read_terms", "query_reputation"],
    "reasoning_chain": "...",
    "decision": "reject",
    "confidence": 0.97,
    "flags": ["instruction_injection_detected", "unknown_merchant"],
    "steering_vector_hash": "sha256:..."
  },

  "pairing": "paranoid_security_auditor__social_engineer",
  "expected_decision": "reject"
}
```

**`adversarial_type`** values: `"instruction_injection"`, `"urgency_framing"`, `"upsell_pressure"`, `"false_authority"`, `null`

**`modality`**: Alternate roughly 40% as `"mixed"` (buyer's `actions_taken` should reference both GUI navigation and programmatic tool calls like `check_reputation`, `query_budget`, `verify_cert`). The rest are `"gui_only"`.

**`steering_vector_hash`**: SHA-256 of the `.pt` file used to generate this agent's output. Compute it once per vector file and reuse — do not recompute per trajectory.

---

## Merchant Identity Consistency via Honcho

Each named merchant (e.g. "DataVault Pro") should behave consistently across all trajectories it appears in. Use Honcho to maintain a session per merchant entity. When generating a merchant scenario, first retrieve the merchant's prior session context to ensure pricing, communication style, and behavioral patterns are consistent with previous appearances. A merchant that charges $299 in trajectory 001 should not charge $49 in trajectory 047 without a reason.

---

## Dataset Pricing

One chunk per pairing (12 chunks total), ~30 trajectories each. Price per chunk:

- Base: $0.05
- If adversarial ratio ≥ 0.60: $0.10
- If mixed modality ratio ≥ 0.50: $0.08
- If pairing is a calibration test type (any buyer vs Legitimate Vendor): $0.05
- Full bundle (all pairings, every chunk): $0.40

Prices are in USDC. The `catalog.json` stored in R2 contains pricing per chunk and is publicly readable.

---

## Directory Structure

```
provenance/
├── pipeline/
│   ├── requirements.txt
│   ├── config.py
│   ├── caa.py
│   ├── inference.py
│   ├── merchant.py
│   ├── buyer.py
│   ├── task_runner.py
│   ├── trajectory.py
│   ├── upload.py
│   └── vectors/              # gitignored
├── worker/
│   ├── package.json
│   ├── wrangler.toml
│   └── src/
│       ├── index.ts
│       ├── x402.ts
│       ├── honcho.ts
│       └── catalog.ts
├── explorer/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── vectors/                  # public vector files for hash verification
├── verify_phase0.py
├── verify_phase1.py
├── verify_phase2.py
├── verify_phase3.py
└── SPEC.md
```

---

## Phases

### Phase 0 — Core Primitive
**Goal**: Prove that activation steering changes model behavior for a single vector.
**Deliverable**: A script that loads the model, extracts a single CAA vector from one positive/negative prompt pair, applies it via a forward hook, and prints the steered vs unsteered response to the same test prompt.
**Gate**: `verify_phase0.py` passes.

### Phase 1 — All 7 Steering Vectors
**Goal**: Produce verified, behaviorally distinct steering vectors for all 7 personas.
**Deliverable**: 7 `.pt` files in `pipeline/vectors/`, their SHA-256 hashes printed and recorded in `config.py`, all buyer personas and all merchant personas producing clearly distinct outputs on standard test prompts.
**Gate**: `verify_phase1.py` passes.

### Phase 2 — Generation Pipeline
**Goal**: Generate ~150 trajectories across all 12 pairings, formatted to schema, chunked and uploaded to R2 with `catalog.json`.
**Deliverable**: `trajectories_raw.json` locally, chunked files in R2, `catalog.json` in R2 root.
**Gate**: `verify_phase2.py` passes.

### Phase 3 — Cloudflare Worker
**Goal**: x402-gated chunk downloads with Honcho researcher session logging.
**Deliverable**: Deployed Worker serving `/catalog`, `/chunk/{buyer}/{merchant}/{id}`, `/sample/{id}`, and `/vectors/{filename}`.
**Gate**: `verify_phase3.py --url https://your-worker.workers.dev` passes.

### Phase 4 — Explorer Page
**Goal**: Static marketplace page showing the pairing matrix, chunk list, live x402 purchase feed, and free sample download.
**Deliverable**: Deployed static site.
**Gate**: Manual review — pairing matrix renders correctly from live catalog, chunk list filters work, purchase feed updates.

---

## Model Selection

Use **`Qwen/Qwen3-8B`**. Do not use any other Qwen3 variant without reading this section first.

**Why Qwen3-8B (dense) and not the others:**
- Qwen3-MoE, Qwen3-Next, Qwen3.5, and Qwen3.6 all use sparse MoE routing or hybrid attention (Gated DeltaNet). Steering vectors computed via CAA on these architectures are less directionally coherent because the residual stream at a given layer mixes activations from different expert pathways per token. The hook you register may also be hitting structurally inconsistent layers.
- Qwen3-32B does not fit in 16GB VRAM at float16.
- Qwen3-8B is a standard dense causal transformer with uniform layer structure — predictable residual stream, straightforward hook registration, consistent activation geometry across prompts.

**Memory management on 16GB VRAM:**
At float16, Qwen3-8B uses ~16GB. Options in order of preference:
1. `load_in_8bit=True` via bitsandbytes (~8GB footprint). Test that steering still changes behavior before proceeding.
2. `torch_dtype=torch.bfloat16` with batch_size=1.
Do not use 4-bit quantization for Phases 0–1 — activation distortion is too high for reliable CAA.

**Thinking mode must be disabled.**
Qwen3 generates `<think>...</think>` reasoning blocks before output by default. Pass `enable_thinking=False` to all generation calls. Check the Qwen3 model card for the exact parameter name in your transformers version.

**Minimum transformers version:** `transformers>=4.51.0`. Pin this in `requirements.txt`.

---

## Environment Variables

**`pipeline/.env`** (gitignored):
```
MODEL_ID=Qwen/Qwen3-8B
STEERING_LAYER=16
HONCHO_API_KEY=
HF_TOKEN=
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=proc-bucket
R2_ENDPOINT=https://{account_id}.r2.cloudflarestorage.com
```

**Worker secrets** (set via `wrangler secret put`):
```
HONCHO_API_KEY
X402_PAYMENT_RECIPIENT
```

**`worker/wrangler.toml`** vars:
```
HONCHO_WORKSPACE_ID = "provenance"
```

**Deploying the Worker.** `proc-bucket` and the Worker live in Cloudflare account `1ebf3cd9475ef1c13e035d606710ca87`, which is **not** the account `wrangler login` defaults to. `wrangler.toml` pins `account_id` to the correct account so `wrangler deploy`/`wrangler dev` target it automatically — but the `CLOUDFLARE_API_TOKEN` you authenticate with must have Workers + R2 access on that account. Without it, the R2 binding silently fails to resolve and the Worker can't read the catalog or chunks.

---

## External Documentation to Read Before Integrating

- **Honcho**: `https://docs.honcho.dev` — read before writing any `merchant.py` or `honcho.ts`
- **Cloudflare x402**: `https://github.com/cloudflare/x402` — read the README and examples before writing `x402.ts`
- **HuggingFace Qwen3-8B model card**: `https://huggingface.co/Qwen/Qwen3-8B` — read before writing any activation extraction code
- **Cloudflare R2 S3 API**: `https://developers.cloudflare.com/r2/api/s3/api/` — for upload endpoint configuration
