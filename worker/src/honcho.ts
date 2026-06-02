/**
 * Researcher-session logging via Honcho v3.
 *
 * Data model:
 *   workspace = HONCHO_WORKSPACE_ID  (must exist in your Honcho account —
 *                                     create at https://app.honcho.dev; v3 does
 *                                     not auto-create workspaces)
 *   session   = the buyer's wallet address (one session per researcher; matches
 *               the "researcher session tracking" intent from SPEC.md)
 *   peer      = the buyer's wallet address (same as session id)
 *   message   = one per purchase, with content + structured metadata
 *
 * Falls back to a no-op when HONCHO_API_KEY is unset so the Worker still
 * functions in local/dev mode without an account.
 */
export interface PurchaseEvent {
  payer: string;
  chunk_id: string;
  pairing: string;
  price_usd: number;
  ts: string;
}

const HONCHO_BASE = "https://api.honcho.dev";

export async function logPurchase(
  env: { HONCHO_API_KEY?: string; HONCHO_WORKSPACE_ID: string },
  evt: PurchaseEvent,
): Promise<void> {
  if (!env.HONCHO_API_KEY) {
    console.log("honcho.logPurchase (no-op, no key):", evt);
    return;
  }

  const workspace = encodeURIComponent(env.HONCHO_WORKSPACE_ID);
  const sessionId = evt.payer; // wallet 0x... satisfies v3's /^[a-zA-Z0-9_-]+$/
  const session = encodeURIComponent(sessionId);
  const headers = {
    "content-type": "application/json",
    "authorization": `Bearer ${env.HONCHO_API_KEY}`,
  };

  // 1. Idempotent get-or-create session. v3 docs don't promise auto-create on
  //    message POST, so doing this explicitly keeps the message call from
  //    failing on first-purchase-for-this-wallet.
  try {
    const sessRes = await fetch(
      `${HONCHO_BASE}/v3/workspaces/${workspace}/sessions`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({ id: sessionId }),
      },
    );
    if (!sessRes.ok) {
      console.warn(
        `honcho session get-or-create non-ok: ${sessRes.status} ${await sessRes.text()}`,
      );
      // Don't return — the message POST may still succeed if Honcho auto-creates.
    }
  } catch (err) {
    console.warn("honcho session get-or-create failed:", err);
  }

  // 2. Post the purchase as a message on the buyer's session.
  try {
    const content = `Purchased ${evt.chunk_id} (${evt.pairing}, $${evt.price_usd.toFixed(2)})`;
    const msgRes = await fetch(
      `${HONCHO_BASE}/v3/workspaces/${workspace}/sessions/${session}/messages`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          messages: [
            {
              content,
              peer_id: evt.payer,
              metadata: { kind: "purchase", ...evt },
            },
          ],
        }),
      },
    );
    if (!msgRes.ok) {
      console.warn(`honcho.logPurchase non-ok: ${msgRes.status} ${await msgRes.text()}`);
    }
  } catch (err) {
    console.warn("honcho.logPurchase failed:", err);
  }
}
