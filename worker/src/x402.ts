/**
 * Build the x402 paymentMiddleware from the live catalog. Each chunk gets
 * its own route entry so prices match catalog.json exactly.
 *
 * Memoized per isolate — the catalog is loaded from R2 on first request and
 * the middleware is constructed once, then reused for subsequent requests.
 */
import type { Context, MiddlewareHandler } from "hono";
import { paymentMiddleware } from "x402-hono";
import { isAddress, type Address } from "viem";
import type { RoutesConfig } from "x402/types";

import type { Catalog } from "./catalog";

const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";

function assertValidRecipient(recipient: string | undefined): asserts recipient is Address {
  if (!recipient) {
    throw new Error("X402_PAYMENT_RECIPIENT is not set — refusing to serve paid routes.");
  }
  if (!isAddress(recipient)) {
    throw new Error(`X402_PAYMENT_RECIPIENT is not a valid EVM address: ${recipient}`);
  }
  if (recipient.toLowerCase() === ZERO_ADDRESS) {
    throw new Error("X402_PAYMENT_RECIPIENT is the zero address — payments would be burned. Set a real wallet via `wrangler secret put X402_PAYMENT_RECIPIENT`.");
  }
}

type Env = {
  DATASET: R2Bucket;
  HONCHO_APP_ID: string;
  HONCHO_API_KEY?: string;
  X402_NETWORK: string;
  X402_FACILITATOR: string;
  X402_PAYMENT_RECIPIENT: string;
};

let cachedMiddleware: MiddlewareHandler | null = null;
let cachedAt = 0;
const TTL_MS = 60_000;

function buildRoutes(catalog: Catalog, network: string): RoutesConfig {
  const routes: RoutesConfig = {};
  for (const chunk of catalog.chunks) {
    const path = `/chunk/${chunk.buyer_persona}/${chunk.merchant_persona}/${chunk.id}`;
    routes[path] = {
      price: `$${chunk.price_usd.toFixed(2)}`,
      network: network as never,
      config: {
        description: `${chunk.pairing} (${chunk.count} trajectories)`,
      },
    };
  }
  return routes;
}

export async function paymentMiddlewareFor(c: Context<{ Bindings: Env }>): Promise<MiddlewareHandler> {
  const now = Date.now();
  if (cachedMiddleware && now - cachedAt < TTL_MS) return cachedMiddleware;

  const env = c.env;
  assertValidRecipient(env.X402_PAYMENT_RECIPIENT);

  const obj = await env.DATASET.get("catalog.json");
  if (!obj) throw new Error("catalog.json not found in R2");
  const catalog = (await obj.json()) as Catalog;

  const routes = buildRoutes(catalog, env.X402_NETWORK);
  cachedMiddleware = paymentMiddleware(
    env.X402_PAYMENT_RECIPIENT,
    routes,
    { url: env.X402_FACILITATOR as `${string}://${string}` },
  );
  cachedAt = now;
  return cachedMiddleware;
}
