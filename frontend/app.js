// Backend URL — set via window.TRIALMATCH_API_BASE in index.html before this
// script loads. Empty string means same-origin (only works if something else
// proxies /quota and /match/stream to the API on this same host).
const API_BASE = window.TRIALMATCH_API_BASE || "";

const form = document.getElementById("match-form");
const textarea = document.getElementById("narrative");
const charCount = document.getElementById("char-count");
const submitBtn = document.getElementById("submit-btn");
const progressEl = document.getElementById("progress");
const resultsEl = document.getElementById("results");
const quotaBar = document.getElementById("quota-bar");

const STAGE_LABELS = {
  extracting: "Extracting patient profile…",
  retrieving: "Retrieving candidate trials (lexical + dense)…",
};

textarea.addEventListener("input", () => {
  charCount.textContent = `${textarea.value.length} / 4000`;
});

async function refreshQuota() {
  try {
    const r = await fetch(`${API_BASE}/quota`);
    const q = await r.json();
    quotaBar.textContent =
      `Live demo budget today: ${q.calls_used_today}/${q.daily_cap} Gemini calls used ` +
      `(${q.remaining} remaining).`;
  } catch {
    quotaBar.textContent = "Demo budget: unavailable.";
  }
}
refreshQuota();
setInterval(refreshQuota, 20000);

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}

function badgeClass(label) {
  if (label === "satisfied") return "sat";
  if (label === "violated") return "vio";
  return "unv";
}

function setProgress(lines) {
  progressEl.hidden = false;
  progressEl.innerHTML = lines
    .map((l) => `<div class="line ${l.active ? "active" : ""}">${escapeHtml(l.text)}</div>`)
    .join("");
}

function renderCriterionRow(row) {
  const evidenceLine =
    row.label === "unverifiable"
      ? `<div class="unv-note">${escapeHtml(row.label_display)}</div>`
      : `<div class="crit-evidence"><span class="tag">patient narrative:</span> "${escapeHtml(row.patient_evidence)}"</div>`;
  return `
    <div class="crit-row">
      <span class="crit-label ${row.label}">${escapeHtml(row.label)}</span>
      <span class="crit-section">(${escapeHtml(row.section)})</span>
      <div class="crit-quote"><span class="tag">criterion:</span> "${escapeHtml(row.criterion_quote)}"</div>
      ${evidenceLine}
      ${row.reasoning ? `<div class="crit-reasoning">${escapeHtml(row.reasoning)}</div>` : ""}
    </div>`;
}

function renderTrialCard(card) {
  const el = document.createElement("details");
  el.className = "trial-card";
  el.id = `trial-${card.nct_id}`;
  el.innerHTML = `
    <summary>
      <div class="trial-head">
        <div>
          <span class="nct">${escapeHtml(card.nct_id)}</span>
          <h3>${escapeHtml(card.title)}</h3>
          <div class="trial-meta">${escapeHtml(card.phase || "")} ${escapeHtml(card.status || "")}</div>
        </div>
      </div>
      <div class="badge-row">
        <span class="badge sat">${card.n_satisfied} satisfied</span>
        <span class="badge vio">${card.n_violated} violated</span>
        <span class="badge unv">${card.n_unverifiable} unverifiable</span>
      </div>
    </summary>
    <div class="criteria-table">
      ${card.criteria.map(renderCriterionRow).join("")}
    </div>`;
  return el;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const narrative = textarea.value.trim();
  if (!narrative) return;

  submitBtn.disabled = true;
  resultsEl.innerHTML = "";
  const cardsByNct = {};
  setProgress([{ text: "Connecting…", active: true }]);

  const url = `${API_BASE}/match/stream?narrative=${encodeURIComponent(narrative)}`;
  const es = new EventSource(url);

  es.addEventListener("stage", (ev) => {
    const { stage } = JSON.parse(ev.data);
    setProgress([{ text: STAGE_LABELS[stage] || stage, active: true }]);
  });

  es.addEventListener("profile", (ev) => {
    const { profile, dropped } = JSON.parse(ev.data);
    const note = profile
      ? "Patient profile extracted."
      : "Extraction failed — falling back to raw narrative for retrieval.";
    setProgress([
      { text: "Extracting patient profile… done", active: false },
      { text: note, active: true },
    ]);
  });

  es.addEventListener("candidates", (ev) => {
    const { nct_ids } = JSON.parse(ev.data);
    setProgress([
      { text: "Retrieving candidate trials… done", active: false },
      { text: `Reasoning over ${nct_ids.length} candidate trials — 0/${nct_ids.length} done`, active: true },
    ]);
    progressEl.dataset.total = nct_ids.length;
    progressEl.dataset.done = "0";
  });

  es.addEventListener("trial_result", (ev) => {
    const card = JSON.parse(ev.data);
    cardsByNct[card.nct_id] = card;
    resultsEl.appendChild(renderTrialCard(card));

    const total = Number(progressEl.dataset.total || "0");
    const done = Number(progressEl.dataset.done || "0") + 1;
    progressEl.dataset.done = String(done);
    setProgress([
      { text: "Retrieving candidate trials… done", active: false },
      { text: `Reasoning over ${total} candidate trials — ${done}/${total} done`, active: done < total },
    ]);
  });

  es.addEventListener("trial_error", (ev) => {
    const { message } = JSON.parse(ev.data);
    const box = document.createElement("div");
    box.className = "error-box";
    box.textContent = `One trial failed and was skipped: ${message}`;
    resultsEl.appendChild(box);
  });

  es.addEventListener("done", (ev) => {
    const { ranked, elapsed_seconds } = JSON.parse(ev.data);
    // Xep lai the theo dung thu tu cuoi cung — trial_result den khong theo thu tu.
    ranked.forEach((nct) => {
      const el = document.getElementById(`trial-${nct}`);
      if (el) resultsEl.appendChild(el);
    });
    setProgress([{ text: `Done in ${elapsed_seconds}s.`, active: false }]);
    es.close();
    submitBtn.disabled = false;
    refreshQuota();
  });

  es.addEventListener("error", (ev) => {
    let message = "Connection lost.";
    try {
      message = JSON.parse(ev.data).message || message;
    } catch {
      /* server-side EventSource errors (e.g. network drop) carry no JSON payload */
    }
    const box = document.createElement("div");
    box.className = "error-box";
    box.textContent = message;
    resultsEl.prepend(box);
    es.close();
    submitBtn.disabled = false;
  });
});
