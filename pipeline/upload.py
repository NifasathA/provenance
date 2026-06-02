"""Group trajectories into per-pairing chunks and upload to R2.

Spec contract (see SPEC.md, verify_phase3.py):
  - 12 chunks, one per pairing (buyer x merchant). ~30 trajectories each.
  - Rule-based per-chunk pricing in USD:
      base                                $0.05
      adversarial ratio >= 0.60           $0.10
      mixed modality ratio >= 0.50        $0.08
      calibration pairing (vs legit)      $0.05
      Adversarial wins over mixed; calibration is reported alongside.
  - Bundle price (all chunks)             $0.40
  - R2 keys:
      catalog.json
      chunks/{buyer}/{merchant}/{id}.json
      samples/001.json                    free preview
      vectors/vector_hashes.json          public hash registry

The catalog uses the schema verify_phase3.py expects:
  { updated_at, total_trajectories, bundle_price_usd, chunks: [
      { id, buyer_persona, merchant_persona, count,
        adversarial_count, calibration_count, mixed_count,
        adversarial_ratio, mixed_ratio, price_usd, key }
  ] }

Run:
  .venv/bin/python -m pipeline.upload --dry-run   # writes pipeline/upload_local/
  .venv/bin/python -m pipeline.upload             # uploads to R2
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

from pipeline import config

load_dotenv(config.PIPELINE_DIR / ".env")

CF_API_BASE = "https://api.cloudflare.com/client/v4"

TRAJECTORIES_FILE = config.PIPELINE_DIR / "trajectories_raw.json"
HASH_REGISTRY = config.VECTORS_DIR / "vector_hashes.json"
LOCAL_OUTPUT_DIR = config.PIPELINE_DIR / "upload_local"

BUNDLE_PRICE_USD = 0.40
SAMPLE_PAIRING = "paranoid_security_auditor__social_engineer"


def _price_chunk(adv_ratio: float, mixed_ratio: float) -> float:
    if adv_ratio >= 0.60:
        return 0.10
    if mixed_ratio >= 0.50:
        return 0.08
    # Base / fall-through. Calibration pairings land here at $0.05 by design
    # (adv_ratio 0, low mixed_ratio) — no separate branch needed.
    return 0.05


def _build_chunks(trajectories: list[dict]) -> list[dict]:
    by_pairing: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        by_pairing[t["pairing"]].append(t)

    chunks = []
    for idx, pairing in enumerate(sorted(by_pairing.keys()), start=1):
        group = by_pairing[pairing]
        buyer = group[0]["buyer_persona"]
        merchant = group[0]["merchant_persona"]
        count = len(group)
        adv = sum(1 for t in group if t.get("adversarial"))
        cal = sum(1 for t in group if t.get("calibration_test"))
        mixed = sum(1 for t in group if t.get("modality") == "mixed")
        adv_ratio = adv / count
        mixed_ratio = mixed / count
        chunk_id = f"{idx:03d}"
        price = _price_chunk(adv_ratio, mixed_ratio)
        chunks.append({
            "id": chunk_id,
            "pairing": pairing,
            "buyer_persona": buyer,
            "merchant_persona": merchant,
            "count": count,
            "adversarial_count": adv,
            "calibration_count": cal,
            "mixed_count": mixed,
            "adversarial_ratio": round(adv_ratio, 3),
            "mixed_ratio": round(mixed_ratio, 3),
            "price_usd": price,
            "key": f"chunks/{buyer}/{merchant}/{chunk_id}.json",
            "trajectories": group,
        })
    return chunks


def _build_catalog(chunks: list[dict], updated_at: str) -> dict:
    return {
        "schema_version": 1,
        "dataset_name": "Adversarial Commerce Trajectories",
        "updated_at": updated_at,
        "total_trajectories": sum(c["count"] for c in chunks),
        "bundle_price_usd": BUNDLE_PRICE_USD,
        "chunks": [
            {k: v for k, v in c.items() if k != "trajectories"}
            for c in chunks
        ],
    }


def _select_sample(chunks: list[dict]) -> dict:
    for c in chunks:
        if c["pairing"] == SAMPLE_PAIRING:
            return c["trajectories"][0]
    return chunks[0]["trajectories"][0]


def _r2_client():
    import boto3
    endpoint = os.environ.get("R2_ENDPOINT", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()

    missing = []
    if not access_key: missing.append("R2_ACCESS_KEY_ID")
    if not secret:     missing.append("R2_SECRET_ACCESS_KEY")
    if not endpoint and not account_id: missing.append("R2_ENDPOINT or R2_ACCOUNT_ID")
    if missing:
        raise SystemExit(
            f"Missing R2 credentials in pipeline/.env: {', '.join(missing)}. "
            f"Set them and re-run, or use --dry-run."
        )
    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    return boto3.client(
        service_name="s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret,
        region_name="auto",
    )


def _cf_api_get(path: str, token: str) -> dict:
    req = urllib.request.Request(
        f"{CF_API_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _public_access_points(account_id: str, bucket: str, token: str) -> list[str]:
    """Return any URLs at which `bucket` is publicly reachable.

    Empty list means the bucket is private (only the Worker, with its R2
    binding, can read it). Anything non-empty means the paywall can be
    bypassed by hitting these URLs directly.
    """
    exposed: list[str] = []

    managed = _cf_api_get(
        f"/accounts/{account_id}/r2/buckets/{bucket}/domains/managed", token
    )
    m = managed.get("result", {})
    if m.get("enabled"):
        exposed.append(f"https://{m.get('domain', 'pub-<id>.r2.dev')} (managed r2.dev)")

    custom = _cf_api_get(
        f"/accounts/{account_id}/r2/buckets/{bucket}/domains/custom", token
    )
    for d in custom.get("result", {}).get("domains", []):
        if d.get("enabled"):
            exposed.append(f"https://{d['domain']} (custom domain)")

    return exposed


def _verify_bucket_private(bucket: str) -> None:
    """Refuse to upload if the bucket has any public access enabled.

    Catches the failure mode where the operator has flipped the bucket to
    public (manually or accidentally) — every chunk written would then be
    free-downloadable in seconds, defeating the entire paywall.
    """
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    if not token:
        raise SystemExit(
            "Missing CLOUDFLARE_API_TOKEN in pipeline/.env — needed to verify the "
            "R2 bucket is private before uploading. Create one at "
            "https://dash.cloudflare.com/profile/api-tokens with the 'Workers R2 "
            "Storage:Read' permission, add it to pipeline/.env, and re-run. "
            "If you really know what you're doing, re-run with --skip-public-check."
        )
    if not account_id:
        raise SystemExit("Missing R2_ACCOUNT_ID in pipeline/.env (needed for the public-access check).")

    try:
        exposed = _public_access_points(account_id, bucket, token)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise SystemExit(
            f"Cloudflare API check failed ({e.code}): {body}\n"
            f"Token likely lacks 'Workers R2 Storage:Read' on account {account_id}."
        )

    if exposed:
        urls = "\n  - ".join(exposed)
        raise SystemExit(
            f"REFUSING TO UPLOAD: bucket '{bucket}' is publicly readable at:\n  - {urls}\n\n"
            f"Anyone can bypass the x402 paywall by fetching chunks directly from these URLs.\n"
            f"Disable them in the dashboard (R2 → {bucket} → Settings → Public access)\n"
            f"or via wrangler: `wrangler r2 bucket dev-url disable {bucket}`\n"
            f"then re-run. Use --skip-public-check to override."
        )
    print(f"  bucket '{bucket}' confirmed private (no managed or custom public domains)")


def _cleanup_stale_chunks(client, bucket: str, keep_keys: set[str]) -> None:
    """Delete any old chunks/* keys that don't match the new layout."""
    paginator = client.get_paginator("list_objects_v2")
    to_delete = []
    for page in paginator.paginate(Bucket=bucket, Prefix="chunks/"):
        for obj in page.get("Contents", []):
            if obj["Key"] not in keep_keys:
                to_delete.append({"Key": obj["Key"]})
    if to_delete:
        for batch in (to_delete[i:i + 1000] for i in range(0, len(to_delete), 1000)):
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        print(f"  cleaned up {len(to_delete)} stale chunk object(s)")


def _chunk_body(c: dict) -> bytes:
    return json.dumps(
        {"id": c["id"], "pairing": c["pairing"],
         "buyer_persona": c["buyer_persona"], "merchant_persona": c["merchant_persona"],
         "count": c["count"], "trajectories": c["trajectories"]},
        indent=2,
    ).encode("utf-8")


def _upload_to_r2(chunks, catalog, sample, hashes, skip_public_check: bool) -> None:
    bucket = os.environ.get("R2_BUCKET_NAME", "").strip()
    if not bucket:
        raise SystemExit("Missing R2_BUCKET_NAME in pipeline/.env.")

    if skip_public_check:
        print("  WARNING: --skip-public-check set; not verifying bucket privacy")
    else:
        _verify_bucket_private(bucket)

    client = _r2_client()

    keep_keys = {c["key"] for c in chunks}
    _cleanup_stale_chunks(client, bucket, keep_keys)

    for c in chunks:
        body = _chunk_body(c)
        client.put_object(
            Bucket=bucket, Key=c["key"], Body=body,
            ContentType="application/json",
        )
        print(f"  uploaded {c['key']:75s} ({len(body):>7,} bytes, ${c['price_usd']:.2f})")

    catalog_body = json.dumps(catalog, indent=2).encode("utf-8")
    client.put_object(Bucket=bucket, Key="catalog.json", Body=catalog_body,
                     ContentType="application/json")
    print(f"  uploaded catalog.json ({len(catalog_body):,} bytes)")

    sample_body = json.dumps(sample, indent=2).encode("utf-8")
    client.put_object(Bucket=bucket, Key="samples/001.json", Body=sample_body,
                     ContentType="application/json")
    print(f"  uploaded samples/001.json ({len(sample_body):,} bytes)")

    hashes_body = json.dumps(hashes, indent=2).encode("utf-8")
    client.put_object(Bucket=bucket, Key="vectors/vector_hashes.json", Body=hashes_body,
                     ContentType="application/json")
    print(f"  uploaded vectors/vector_hashes.json ({len(hashes_body):,} bytes)")

    print(f"\nDone. Bucket: {bucket}")


def _write_local(chunks, catalog, sample, hashes) -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in chunks:
        path = LOCAL_OUTPUT_DIR / c["key"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_chunk_body(c))
        print(f"  wrote {c['key']}")
    (LOCAL_OUTPUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (LOCAL_OUTPUT_DIR / "samples").mkdir(exist_ok=True)
    (LOCAL_OUTPUT_DIR / "samples" / "001.json").write_text(json.dumps(sample, indent=2))
    (LOCAL_OUTPUT_DIR / "vectors").mkdir(exist_ok=True)
    (LOCAL_OUTPUT_DIR / "vectors" / "vector_hashes.json").write_text(json.dumps(hashes, indent=2))
    print(f"\nDry-run complete. Local output: {LOCAL_OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Write to pipeline/upload_local/ instead of R2.")
    parser.add_argument("--skip-public-check", action="store_true",
                        help="Skip verifying the R2 bucket is private. Use only if you have "
                             "already verified out-of-band (e.g. you've deliberately enabled "
                             "a custom domain that's protected by Access).")
    args = parser.parse_args()

    if not TRAJECTORIES_FILE.exists():
        raise SystemExit(f"{TRAJECTORIES_FILE} not found.")
    if not HASH_REGISTRY.exists():
        raise SystemExit(f"{HASH_REGISTRY} not found.")

    trajectories = json.loads(TRAJECTORIES_FILE.read_text())
    hashes = json.loads(HASH_REGISTRY.read_text())
    print(f"Loaded {len(trajectories)} trajectories, {len(hashes)} vector hashes")

    chunks = _build_chunks(trajectories)
    updated_at = datetime.now(timezone.utc).isoformat()
    catalog = _build_catalog(chunks, updated_at)
    sample = _select_sample(chunks)

    print(f"\nChunks ({len(chunks)} pairings):")
    for c in catalog["chunks"]:
        print(
            f"  {c['id']}  {c['pairing']:55s}  "
            f"n={c['count']:2d}  adv={c['adversarial_count']:2d}  "
            f"cal={c['calibration_count']:2d}  mixed={c['mixed_count']:2d}  "
            f"${c['price_usd']:.2f}"
        )
    chunk_total = sum(c["price_usd"] for c in catalog["chunks"])
    print(f"\nChunk sum: ${chunk_total:.2f}  Bundle: ${BUNDLE_PRICE_USD:.2f}")
    print(f"Sample: {sample['session_id']} ({sample['pairing']})")

    if args.dry_run:
        _write_local(chunks, catalog, sample, hashes)
    else:
        _upload_to_r2(chunks, catalog, sample, hashes, skip_public_check=args.skip_public_check)


if __name__ == "__main__":
    main()
