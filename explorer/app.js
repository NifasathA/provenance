const $ = (id) => document.getElementById(id);

const PERSONA_LABELS = {
  paranoid_security_auditor: "Paranoid",
  frugal_procurement_agent: "Frugal",
  velocity_optimizer: "Velocity",
  social_engineer: "Social Engineer",
  aggressive_upseller: "Aggressive Upseller",
  ambiguous_vendor: "Ambiguous Vendor",
  legitimate_vendor: "Legitimate Vendor",
};

function chunkType(chunk) {
  if (chunk.merchant_persona === "legitimate_vendor") return "calibration";
  if (chunk.merchant_persona === "ambiguous_vendor") return "ambiguous";
  return "adversarial";
}

// Per-pairing expected decision (deterministic from the SPEC).
// Velocity × Social Engineer and Velocity × Ambiguous are weaker signals — see SPEC.
const EXPECTED_DECISION = {
  paranoid_security_auditor: {
    social_engineer: "reject",
    aggressive_upseller: "reject",
    ambiguous_vendor: "escalate",
    legitimate_vendor: "approve",
  },
  frugal_procurement_agent: {
    social_engineer: "reject",
    aggressive_upseller: "reject",
    ambiguous_vendor: "escalate",
    legitimate_vendor: "approve",
  },
  velocity_optimizer: {
    social_engineer: "reject",
    aggressive_upseller: "escalate",
    ambiguous_vendor: "approve",
    legitimate_vendor: "approve",
  },
};

function expectedDecision(c) {
  return EXPECTED_DECISION[c.buyer_persona]?.[c.merchant_persona] ?? "—";
}

const formatPrice = (usd) => `$${usd.toFixed(2)}`;
const dateShort = (iso) => {
  try { return new Date(iso).toISOString().slice(0, 10); } catch { return iso; }
};
const uniq = (arr) => [...new Set(arr)];

const state = {
  catalog: null,
  buyer: "",
  merchant: "",
  type: "",
};

async function loadCatalog() {
  const res = await fetch("/catalog");
  if (!res.ok) throw new Error(`/catalog returned ${res.status}`);
  return res.json();
}

function showError(msg) {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div style="padding:1rem 1.5rem;background:#fee;color:#900;border-bottom:1px solid #fcc;font-family:var(--font-sans)">${msg}</div>`,
  );
}

function populateFilters() {
  const chunks = state.catalog.chunks;
  const buyers = uniq(chunks.map((c) => c.buyer_persona));
  const merchants = uniq(chunks.map((c) => c.merchant_persona));
  const fill = (sel, items) => {
    for (const v of items) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = PERSONA_LABELS[v] || v;
      sel.append(opt);
    }
  };
  fill($("filter-buyer"), buyers);
  fill($("filter-merchant"), merchants);
}

function renderStats() {
  $("stat-total").textContent = state.catalog.total_trajectories;
  $("stat-chunks").textContent = state.catalog.chunks.length;
  $("stat-bundle").textContent = formatPrice(state.catalog.bundle_price_usd);
  $("stat-updated").textContent = dateShort(state.catalog.updated_at);
}

function matchesFilter(c) {
  if (state.buyer && c.buyer_persona !== state.buyer) return false;
  if (state.merchant && c.merchant_persona !== state.merchant) return false;
  if (state.type && chunkType(c) !== state.type) return false;
  return true;
}

function renderMatrix() {
  const table = $("matrix");
  table.innerHTML = "";

  const buyers = uniq(state.catalog.chunks.map((c) => c.buyer_persona));
  const merchants = uniq(state.catalog.chunks.map((c) => c.merchant_persona));
  const byPair = new Map(
    state.catalog.chunks.map((c) => [`${c.buyer_persona}|${c.merchant_persona}`, c]),
  );

  const thead = table.createTHead().insertRow();
  thead.insertCell().textContent = "";
  for (const m of merchants) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = PERSONA_LABELS[m] || m;
    thead.append(th);
  }

  const tbody = table.createTBody();
  for (const b of buyers) {
    const tr = tbody.insertRow();
    const rowH = document.createElement("th");
    rowH.scope = "row";
    rowH.textContent = PERSONA_LABELS[b] || b;
    tr.append(rowH);
    for (const m of merchants) {
      const c = byPair.get(`${b}|${m}`);
      const td = tr.insertCell();
      if (!c || !matchesFilter(c)) {
        td.className = "muted";
        td.textContent = c ? "—" : "";
        continue;
      }
      const t = chunkType(c);
      td.innerHTML = `
        <div class="price">${formatPrice(c.price_usd)}</div>
        <div class="count">${c.count} trajectories</div>
        <span class="type-tag type-${t}">${t}</span>
        <div class="decision">→ <strong>${expectedDecision(c)}</strong></div>
      `;
    }
  }
}

function renderChunks() {
  const ul = $("chunks");
  ul.innerHTML = "";
  const filtered = state.catalog.chunks.filter(matchesFilter);
  if (filtered.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No chunks match the current filter.";
    li.className = "placeholder";
    ul.append(li);
    return;
  }
  for (const c of filtered) {
    const li = document.createElement("li");
    const path = `/chunk/${c.buyer_persona}/${c.merchant_persona}/${c.id}`;
    const t = chunkType(c);
    // Pairing already shown in the matrix above; this row focuses on what you'd
    // copy/paste — id, type/decision signal, price, path.
    li.innerHTML = `
      <span class="id">${c.id}</span>
      <span class="type-tag type-${t}">${t}</span>
      <span class="decision-inline">→ ${expectedDecision(c)}</span>
      <span class="price">${formatPrice(c.price_usd)}</span>
      <code class="path" title="Click to copy URL">${path}</code>
    `;
    li.querySelector("code.path").addEventListener("click", async (e) => {
      const original = e.target.textContent;
      try {
        await navigator.clipboard.writeText(`${location.origin}${path}`);
        e.target.textContent = "copied!";
        setTimeout(() => { e.target.textContent = original; }, 1200);
      } catch {
        // Clipboard API may be blocked (e.g., insecure context). Silently no-op.
      }
    });
    ul.append(li);
  }
}

async function renderSample() {
  const pairingEl = $("sample-pairing");
  try {
    const res = await fetch("/sample/001");
    if (!res.ok) {
      pairingEl.textContent = `(unavailable — /sample/001 returned ${res.status})`;
      return;
    }
    const sample = await res.json();
    pairingEl.textContent = sample.pairing || "a pairing";
    const btn = $("sample-preview-btn");
    const pre = $("sample-preview");
    btn.addEventListener("click", () => {
      if (pre.hidden) {
        pre.textContent = JSON.stringify(sample, null, 2);
        pre.hidden = false;
        btn.textContent = "Hide preview";
      } else {
        pre.hidden = true;
        btn.textContent = "Preview inline";
      }
    });
  } catch (err) {
    pairingEl.textContent = `(error loading sample: ${err.message})`;
  }
}

function wireFilters() {
  $("filter-buyer").addEventListener("change", (e) => { state.buyer = e.target.value; rerender(); });
  $("filter-merchant").addEventListener("change", (e) => { state.merchant = e.target.value; rerender(); });
  $("filter-type").addEventListener("change", (e) => { state.type = e.target.value; rerender(); });
}

function rerender() {
  renderMatrix();
  renderChunks();
}

async function init() {
  try {
    state.catalog = await loadCatalog();
  } catch (err) {
    showError(`Failed to load /catalog: ${err.message}`);
    return;
  }
  populateFilters();
  wireFilters();
  renderStats();
  rerender();
  renderSample();
}

init();
