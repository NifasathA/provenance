export interface CatalogChunk {
  id: string;
  pairing: string;
  buyer_persona: string;
  merchant_persona: string;
  count: number;
  adversarial_count: number;
  calibration_count: number;
  mixed_count: number;
  adversarial_ratio: number;
  mixed_ratio: number;
  price_usd: number;
  key: string;
}

export interface Catalog {
  schema_version: number;
  dataset_name: string;
  updated_at: string;
  total_trajectories: number;
  bundle_price_usd: number;
  chunks: CatalogChunk[];
}

export async function loadCatalog(bucket: R2Bucket): Promise<Catalog> {
  const obj = await bucket.get("catalog.json");
  if (!obj) throw new Error("catalog.json missing from R2");
  return (await obj.json()) as Catalog;
}

export async function readJson(bucket: R2Bucket, key: string): Promise<unknown> {
  const obj = await bucket.get(key);
  if (!obj) return null;
  return await obj.json();
}

export function findChunk(
  catalog: Catalog,
  buyer: string,
  merchant: string,
  id: string,
): CatalogChunk | undefined {
  return catalog.chunks.find(
    (c) => c.buyer_persona === buyer && c.merchant_persona === merchant && c.id === id,
  );
}
