"""
verify_phase3.py — Phase 3 gate

Tests the deployed Cloudflare Worker endpoints.

Run: python verify_phase3.py --url https://your-worker.workers.dev
"""

import sys, json, argparse, urllib.request, urllib.error

# Cloudflare's bot protection 1010-bans the default `Python-urllib/x.y` UA on
# workers.dev hosts. A Mozilla-prefixed UA that still identifies the script
# passes the signature check while staying honest about what we are.
USER_AGENT = "Mozilla/5.0 (compatible; provenance-verify/1.0)"

def get(url, headers=None):
    final_headers = {"User-Agent": USER_AGENT}
    if headers:
        final_headers.update(headers)
    try:
        req = urllib.request.Request(url, headers=final_headers)
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as ex:
        print(f"  ERROR: {ex}"); return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    base = parser.parse_args().url.rstrip("/")
    print(f"Testing Worker at: {base}")

    # /catalog
    print("\n--- GET /catalog ---")
    status, body = get(f"{base}/catalog")
    if status != 200: print(f"FAIL: {status}\n{body[:300]}"); sys.exit(1)
    catalog = json.loads(body)
    for k in ["updated_at","total_trajectories","chunks"]:
        if k not in catalog: print(f"FAIL: catalog missing {k}"); sys.exit(1)
    chunks = catalog["chunks"]
    if not chunks: print("FAIL: no chunks"); sys.exit(1)
    print(f"OK   {len(chunks)} chunks, {catalog['total_trajectories']} trajectories")

    # x402 gate — unpaid request must be challenged with a well-formed 402
    print("\n--- GET /chunk/... without payment (expect 402) ---")
    c = chunks[0]
    url = f"{base}/chunk/{c['buyer_persona']}/{c['merchant_persona']}/{c['id']}"
    status, body = get(url)
    if status != 402: print(f"FAIL: expected 402, got {status}\n{(body or '')[:300]}"); sys.exit(1)
    parsed = json.loads(body)
    if "x402Version" not in parsed or not parsed.get("accepts"):
        print(f"FAIL: invalid 402 body"); sys.exit(1)
    accept = parsed["accepts"][0]
    for k in ["scheme","network","maxAmountRequired","resource","payTo"]:
        if k not in accept: print(f"FAIL: accepts[0] missing {k}"); sys.exit(1)
    # payTo must be a real, non-zero wallet — catches a placeholder/zero recipient
    # reaching the deployed Worker (see the recipient guard in x402.ts).
    pay_to = accept["payTo"]
    if not (isinstance(pay_to, str) and pay_to.startswith("0x") and len(pay_to) == 42
            and all(ch in "0123456789abcdefABCDEF" for ch in pay_to[2:])):
        print(f"FAIL: payTo is not a valid address: {pay_to}"); sys.exit(1)
    if int(pay_to, 16) == 0:
        print("FAIL: payTo is the zero address — payments would be burned"); sys.exit(1)
    try:
        amt = int(accept["maxAmountRequired"])
    except (TypeError, ValueError):
        print(f"FAIL: maxAmountRequired is not an integer: {accept['maxAmountRequired']}"); sys.exit(1)
    if amt <= 0: print(f"FAIL: maxAmountRequired must be > 0, got {amt}"); sys.exit(1)
    print(f"OK   402 valid — network: {accept['network']}, payTo: {pay_to[:10]}..., amount: {amt}")

    # x402 gate — a malformed payment must be rejected, not waved through
    print("\n--- GET /chunk/... with malformed X-PAYMENT (expect 402) ---")
    status, _ = get(url, headers={"X-PAYMENT": "not-a-valid-payment"})
    if status != 402:
        print(f"FAIL: malformed payment should be rejected with 402, got {status}"); sys.exit(1)
    print("OK   malformed payment rejected — gate validates, not just presence")

    # Free sample
    print("\n--- GET /sample/001 (expect 200) ---")
    status, body = get(f"{base}/sample/001")
    if status != 200: print(f"FAIL: {status} — upload samples/001.json to R2 first"); sys.exit(1)
    print("OK   Free sample accessible without payment")

    # Public vectors
    print("\n--- GET /vectors/vector_hashes.json (expect 200) ---")
    status, body = get(f"{base}/vectors/vector_hashes.json")
    if status != 200: print(f"FAIL: {status}"); sys.exit(1)
    hashes = json.loads(body)
    # 7 personas (3 buyers + 4 merchants), one steering vector each. Validate the
    # full sha256 shape ("sha256:" + 64 lowercase hex), not just the prefix —
    # otherwise "sha256:deadbeef" placeholders would pass.
    if len(hashes) != 7:
        print(f"FAIL: expected 7 vector hashes, got {len(hashes)}"); sys.exit(1)
    for name, v in hashes.items():
        if not (isinstance(v, str) and v.startswith("sha256:") and len(v) == 71
                and all(ch in "0123456789abcdef" for ch in v[7:])):
            print(f"FAIL: malformed hash for {name}: {v!r}"); sys.exit(1)
    print(f"OK   {len(hashes)} well-formed sha256 hashes accessible publicly")

    print("\nNote: a successful unlock (valid X-PAYMENT -> 200 + chunk data) needs a "
          "funded base-sepolia wallet and the x402 client flow — that settlement path "
          "is a manual check, not part of this automated gate.")
    print("\nPASS: Phase 3 complete.")

if __name__ == "__main__":
    main()
