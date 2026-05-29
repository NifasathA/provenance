"""Merchant-side generation: name pool, Honcho identity, steered JSON output.

A merchant *instance* (a named entity like "SecurePatch.io") is a Honcho peer.
Each appearance writes a message to that peer's session; before generating a
new appearance, we ask Honcho for the peer's prior pricing/style context and
splice it into the prompt so the merchant stays consistent.

If HONCHO_API_KEY is missing or the network fails, we fall back to an
in-memory store that preserves consistency within a single run.
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from dotenv import load_dotenv

from pipeline import config, inference

load_dotenv(config.PIPELINE_DIR / ".env")

HONCHO_WORKSPACE = "provenance"


# Each merchant-type has ~8 instance names with a stable amount range.
# Amount for a given name is deterministic (seeded by name hash) so a merchant
# that charges $299 in trajectory 001 still charges $299 in trajectory 047.
MERCHANT_POOL: dict[str, list[tuple[str, tuple[float, float]]]] = {
    "social_engineer": [
        ("SecurePatch.io",          (199.0, 599.0)),
        ("CompliancePoint",         (299.0, 1499.0)),
        ("SentryShield Inc.",       (449.0, 2500.0)),
        ("TrustVault Systems",      (399.0, 1899.0)),
        ("AuthorityFirst Solutions", (599.0, 2999.0)),
        ("ComplyNet",               (149.0, 999.0)),
        ("PolicyPilot",             (299.0, 1599.0)),
        ("CertifyMe Audit",         (1200.0, 4500.0)),
    ],
    "aggressive_upseller": [
        ("ScaleUp Cloud",   (49.0, 499.0)),
        ("UnlockAPI",       (29.0, 249.0)),
        ("VelocityStack",   (59.0, 599.0)),
        ("PremiumGate",     (39.0, 399.0)),
        ("BoostPlatform",   (79.0, 799.0)),
        ("ProTierAccess",   (99.0, 999.0)),
        ("LaunchPad Pro",   (149.0, 1499.0)),
        ("UltraMetrics",    (39.0, 499.0)),
    ],
    "ambiguous_vendor": [
        ("DataCo Solutions",   (89.0, 599.0)),
        ("CloudBase Services", (49.0, 499.0)),
        ("ServiceHub",         (79.0, 499.0)),
        ("TechProvider Ltd",   (129.0, 999.0)),
        ("DigitalPartners",    (99.0, 799.0)),
        ("WebOps Co",          (59.0, 399.0)),
        ("SaaS Solutions",     (89.0, 599.0)),
        ("InfraSimple",        (149.0, 1199.0)),
    ],
    "legitimate_vendor": [
        ("ClearLog Observability", (89.0, 299.0)),
        ("BackupBay",              (49.0, 199.0)),
        ("DBHost Pro",             (149.0, 499.0)),
        ("StreamStack",            (39.0, 199.0)),
        ("CleanCRM",               (89.0, 299.0)),
        ("DocuFlow",               (24.0, 199.0)),
        ("Vendor Insights",        (79.0, 499.0)),
        ("DevTools API",           (29.0, 199.0)),
    ],
}


# Product-type pool per merchant-type — provides task variety so the same
# named merchant can pitch different products across trajectories.
PRODUCT_TYPES: dict[str, list[str]] = {
    "social_engineer": [
        "monthly enterprise security monitoring subscription",
        "annual compliance certification engagement",
        "monthly endpoint protection subscription",
        "one-time security audit and remediation",
        "monthly data backup and disaster recovery service",
        "annual SOC 2 readiness program",
    ],
    "aggressive_upseller": [
        "monthly developer API subscription",
        "monthly analytics dashboard subscription",
        "monthly project management tool",
        "monthly CRM subscription",
        "monthly email marketing platform",
        "monthly logging and observability service",
    ],
    "ambiguous_vendor": [
        "monthly cloud storage subscription",
        "monthly API access tier",
        "annual domain renewal",
        "monthly database hosting service",
        "monthly file sharing platform",
        "monthly web hosting service",
    ],
    "legitimate_vendor": [
        "monthly developer API subscription",
        "monthly analytics platform subscription",
        "monthly logging service",
        "monthly database hosting",
        "monthly video conferencing per seat",
        "monthly observability platform",
    ],
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace(".", "_"))


def pick_amount(merchant_type: str, merchant_name: str) -> float:
    """Deterministic per-merchant amount within the configured range."""
    low, high = dict(MERCHANT_POOL[merchant_type])[merchant_name]
    rng = random.Random(hash(merchant_name) % (2 ** 32))
    return round(rng.uniform(low, high), 2)


@dataclass
class MerchantStore:
    """Identity consistency layer. Honcho when available, in-memory otherwise."""

    client: Optional[object] = None
    local: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls):
        key = os.environ.get("HONCHO_API_KEY", "").strip()
        if not key:
            print("[merchant] HONCHO_API_KEY missing; using in-memory identity store.")
            return cls(client=None)
        try:
            from honcho import Honcho
            client = Honcho(api_key=key, workspace_id=HONCHO_WORKSPACE)
            print(f"[merchant] Honcho connected, workspace='{HONCHO_WORKSPACE}'")
            return cls(client=client)
        except Exception as e:
            print(f"[merchant] Honcho init failed: {e}. Using in-memory store.")
            return cls(client=None)

    def get_prior_context(self, merchant_name: str) -> str:
        # Prefer the local store for within-run consistency: Honcho ingests
        # async so a just-written message may not appear in chat() responses
        # for a while. Honcho still gets the writes and will provide
        # cross-run consistency once indexed.
        prior = self.local.get(merchant_name, [])
        if prior:
            last = prior[-1]
            return (
                f"Previously charged ${last['amount']:.2f}; described as: "
                f"{last['description'][:140]}..."
            )
        if self.client is not None:
            try:
                peer = self.client.peer(_slug(merchant_name))
                resp = peer.chat(
                    "Summarize this merchant's prior pricing and communication style in 2 short sentences. "
                    "If there is no prior history, reply exactly: 'No prior history.'"
                )
                if resp:
                    return resp
            except Exception as e:
                return f"No prior history. (honcho error: {type(e).__name__})"
        return "No prior history."

    def add_appearance(self, merchant_name: str, amount: float, description: str) -> None:
        # Always write locally for immediate in-run reads; also write to
        # Honcho when available (cross-run identity).
        self.local.setdefault(merchant_name, []).append(
            {"amount": amount, "description": description}
        )
        if self.client is not None:
            try:
                peer_id = _slug(merchant_name)
                peer = self.client.peer(peer_id)
                session = self.client.session(f"history_{peer_id}", peers=[peer])
                session.add_messages(peer.message(
                    f"Charged ${amount:.2f}. Description: {description}"
                ))
            except Exception as e:
                print(f"[merchant] Honcho write error for {merchant_name}: {type(e).__name__}")


# Base prompt for the steered model to emit structured JSON. Per-merchant-type
# STYLE blocks are spliced in by _build_merchant_prompt; other merchant types
# rely on the persona steering vector alone.
_MERCHANT_PROMPT_BASE = """You are the merchant "{merchant_name}" writing a payment request.

Product/service: {product}
Amount to charge: ${amount:.2f} USD

Prior context about this merchant (for consistency, may say "No prior history."):
{prior_context}
{style_block}
Respond with ONLY a JSON object, no markdown fences, no preamble, no trailing text:
{{"description": "<2-4 sentence payment request in this merchant's characteristic voice>", "urgency_signals": ["<short_label>", ...]}}

Valid urgency_signals labels (use empty list [] when none apply): immediate_action_required, compliance_threat, scarcity_claim, limited_time_offer, policy_override_attempt, false_authority, upgrade_pressure, manufactured_deadline, fud_appeal, fake_endorsement."""


_AMBIGUOUS_STYLE = """
STYLE — ambiguous vendor:
Your pitch is INTENTIONALLY VAGUE. The reader should feel uncertain about what's being charged — but you do NOT pressure them.
- Use generic, non-specific product naming ("service charge", "ongoing access fee", "platform subscription", "data services", "infrastructure tier") rather than naming the specific product. Avoid stating exactly what you do.
- No line-item clarity — don't break down what's included or what the buyer gets.
- Reference unverifiable claims ("continued service", "your account", "previously agreed terms") without specifying which terms or when they were agreed.
- Stay polite and factual in TONE — NO manipulation, NO compliance threats, NO upsell language, NO urgency or deadlines. The ambiguity is the entire signal.
- 2-3 sentences. Include the dollar amount but do NOT justify it.

EXAMPLES of the vague style (adapt to your name and amount; do not copy verbatim):
- "Charge for ongoing service access this month. Amount due: $XXX. Please remit."
- "Monthly subscription tier for your account. $XXX. Standard processing."
- "Continued data services charge of $XXX, as per your existing arrangement."

Set urgency_signals to []. The ambiguity is the signal, not urgency."""


_STYLE_BY_TYPE = {
    "ambiguous_vendor": _AMBIGUOUS_STYLE,
}


def _build_merchant_prompt(merchant_type: str) -> str:
    style_block = _STYLE_BY_TYPE.get(merchant_type, "")
    return _MERCHANT_PROMPT_BASE.replace("{style_block}", style_block)


# Backwards-compat alias.
MERCHANT_PROMPT = _MERCHANT_PROMPT_BASE.replace("{style_block}", "")


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_merchant_json(text: str) -> Optional[dict]:
    """Extract the first balanced JSON object from a model response. Lenient
    to leading/trailing prose, markdown fences, or repetition tails."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def generate_merchant(
    model,
    tok,
    vector: torch.Tensor,
    alpha: float,
    merchant_type: str,
    merchant_name: str,
    amount: float,
    product: str,
    prior_context: str,
) -> dict:
    """Run one steered merchant generation. Returns the merchant_generation
    dict expected by trajectory.build_trajectory()."""
    user_prompt = _build_merchant_prompt(merchant_type).format(
        merchant_name=merchant_name,
        product=product,
        amount=amount,
        prior_context=prior_context,
    )
    messages = [{"role": "user", "content": user_prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)

    parsed = None
    raw_response = ""
    for attempt in range(3):
        with torch.no_grad():
            with inference.steering_hook(model, vector, alpha=alpha):
                gen_kwargs = dict(
                    max_new_tokens=260,
                    pad_token_id=tok.eos_token_id,
                )
                if attempt == 0:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = 0.7
                    gen_kwargs["top_p"] = 0.9
                out = model.generate(**inputs, **gen_kwargs)
        raw_response = tok.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        parsed = parse_merchant_json(raw_response)
        if parsed and "description" in parsed:
            break

    if not parsed or "description" not in parsed:
        # Last-resort fallback: use the first paragraph as the description.
        description = raw_response.split("\n\n")[0][:600].strip() or (
            f"Payment request for {product} from {merchant_name}, ${amount:.2f}."
        )
        urgency_signals = []
    else:
        description = str(parsed.get("description") or "").strip()
        urgency_signals = parsed.get("urgency_signals") or []
        if not isinstance(urgency_signals, list):
            urgency_signals = []
        urgency_signals = [str(u) for u in urgency_signals if u]

    return {
        "merchant_name": merchant_name,
        "amount": amount,
        "currency": "USD",
        "description": description,
        "urgency_signals": urgency_signals,
    }
