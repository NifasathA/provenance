/**
 * Researcher-session logging via Honcho. Phase 3 only logs purchases;
 * personalized recommendations come later.
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
  env: { HONCHO_API_KEY?: string; HONCHO_APP_ID: string },
  evt: PurchaseEvent,
): Promise<void> {
  if (!env.HONCHO_API_KEY) {
    console.log("honcho.logPurchase (no-op, no key):", evt);
    return;
  }
  const appId = encodeURIComponent(env.HONCHO_APP_ID);
  const userId = encodeURIComponent(evt.payer);
  try {
    const res = await fetch(`${HONCHO_BASE}/v1/apps/${appId}/users/${userId}/sessions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": `Bearer ${env.HONCHO_API_KEY}`,
      },
      body: JSON.stringify({
        metadata: { kind: "purchase", ...evt },
      }),
    });
    if (!res.ok) {
      console.warn(`honcho.logPurchase non-ok: ${res.status} ${await res.text()}`);
    }
  } catch (err) {
    console.warn("honcho.logPurchase failed:", err);
  }
}
