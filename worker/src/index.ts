import { Hono } from "hono";
import { exact } from "x402/schemes";

import { findChunk, loadCatalog, readJson } from "./catalog";
import type { Env } from "./env";
import { logPurchase } from "./honcho";
import { paymentMiddlewareFor } from "./x402";

const app = new Hono<{ Bindings: Env }>();

// `/` is served by the explorer's index.html via wrangler [assets]; only
// unmatched paths reach the Worker, so no Hono `/` route is needed here.

// Free: full catalog.
app.get("/catalog", async (c) => {
  const catalog = await loadCatalog(c.env.DATASET);
  return c.json(catalog);
});

// Free: a single sample trajectory.
app.get("/sample/:id", async (c) => {
  const id = c.req.param("id");
  const data = await readJson(c.env.DATASET, `samples/${id}.json`);
  if (!data) return c.notFound();
  return c.json(data);
});

// Free: vector hash registry (and any other public vectors/ asset).
app.get("/vectors/:filename", async (c) => {
  const filename = c.req.param("filename");
  const data = await readJson(c.env.DATASET, `vectors/${filename}`);
  if (!data) return c.notFound();
  return c.json(data);
});

// Paid: chunk download. paymentMiddlewareFor returns the cached x402-hono
// middleware whose routes mirror catalog.json prices.
app.use("/chunk/:buyer/:merchant/:id", async (c, next) => {
  const middleware = await paymentMiddlewareFor(c);
  return middleware(c, next);
});

app.get("/chunk/:buyer/:merchant/:id", async (c) => {
  const { buyer, merchant, id } = c.req.param();
  const catalog = await loadCatalog(c.env.DATASET);
  const chunk = findChunk(catalog, buyer, merchant, id);
  if (!chunk) return c.notFound();

  const obj = await c.env.DATASET.get(chunk.key);
  if (!obj) {
    // Catalogued chunk, missing R2 object = server-side drift, not a bad URL.
    // 503 (status >=400) makes x402-hono skip settle(), so the buyer isn't charged.
    console.error(`catalog/R2 drift: chunk ${chunk.id} references missing object ${chunk.key}`);
    return c.json(
      {
        error:
          "Chunk is catalogued but its data object is missing from storage. You were not charged; please retry shortly.",
        chunk_id: chunk.id,
      },
      503,
    );
  }

  // x402-hono's middleware accepted the request, so X-PAYMENT is present and
  // decodable. Pull the EIP-3009 signer (`authorization.from`) — that's the
  // wallet that actually paid, and the only stable id we can group by.
  let payer = "unknown";
  try {
    const decoded = exact.evm.decodePayment(c.req.header("X-PAYMENT") ?? "");
    if (decoded.payload && "authorization" in decoded.payload) {
      payer = decoded.payload.authorization.from;
    }
  } catch {
    // Middleware would have rejected an undecodable payment, so this is
    // defensive only — log under "unknown" rather than fail the download.
  }

  c.executionCtx.waitUntil(
    logPurchase(c.env, {
      payer,
      chunk_id: chunk.id,
      pairing: chunk.pairing,
      price_usd: chunk.price_usd,
      ts: new Date().toISOString(),
    }),
  );

  return new Response(obj.body, {
    headers: { "content-type": "application/json" },
  });
});

export default app;
