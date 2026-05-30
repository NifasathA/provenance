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
import { SupportedEVMNetworks, type Network, type RoutesConfig } from "x402/types";

import { loadCatalog, type Catalog } from "./catalog";
import type { Env } from "./env";

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

function assertSupportedNetwork(network: string): asserts network is Network {
  if (!SupportedEVMNetworks.some((n) => n === network)) {
    throw new Error(
      `X402_NETWORK "${network}" is not a supported EVM network. Expected one of: ${SupportedEVMNetworks.join(", ")}.`,
    );
  }
}

// Per-isolate cache of the built middleware: the catalog is read from R2 and the
// middleware constructed once, then reused for up to TTL_MS. Consequence — the
// gate's routes/prices can lag a catalog change by up to TTL_MS. The chunk
// handler (index.ts) loads the catalog fresh each request, so within that window
// the two can diverge: a just-added chunk may serve free until the gate
// refreshes, a just-removed chunk gets gated then 404s, a repriced chunk charges
// the old price. Acceptable because the catalog only changes on a manual
// re-upload; wait out TTL_MS after uploading before treating new prices as live.
let cachedMiddleware: MiddlewareHandler | null = null;
let cachedAt = 0;
const TTL_MS = 60_000;

function buildRoutes(catalog: Catalog, network: Network): RoutesConfig {
  const routes: RoutesConfig = {};
  for (const chunk of catalog.chunks) {
    const path = `/chunk/${chunk.buyer_persona}/${chunk.merchant_persona}/${chunk.id}`;
    routes[path] = {
      price: `$${chunk.price_usd.toFixed(2)}`,
      network,
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
  assertSupportedNetwork(env.X402_NETWORK);

  const catalog = await loadCatalog(env.DATASET);
  const routes = buildRoutes(catalog, env.X402_NETWORK);
  cachedMiddleware = paymentMiddleware(
    env.X402_PAYMENT_RECIPIENT,
    routes,
    { url: env.X402_FACILITATOR as `${string}://${string}` },
  );
  cachedAt = now;
  return cachedMiddleware;
}
