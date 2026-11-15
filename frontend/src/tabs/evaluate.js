import { getEvaluation, listEvaluations, startEvaluation } from "../api.js";

let meta = null;
let selected = new Set();
let polling = null;

function renderVariants() {
  const container = document.getElementById("eval-variants");
  container.replaceChildren();
  selected = new Set();
  for (const variant of meta.variants) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = variant.id;
    checkbox.checked = true;
    if (checkbox.checked) selected.add(variant.id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(variant.id);
      else selected.delete(variant.id);
    });
    label.append(checkbox, ` ${variant.label}`);
    container.append(label);
  }
}

const pct = (value) => `${(value * 100).toFixed(1)}%`;

function renderBreakdown(run, container) {
  if (!run.metrics) return;
  const heading = document.createElement("h3");
  heading.textContent = "Per-item accuracy";
  container.append(heading);
  for (const [variant, metrics] of Object.entries(run.metrics)) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = variant;
    details.append(summary);

    const table = document.createElement("table");
    const head = document.createElement("tr");
    for (const label of ["Item", "N", "Top-1", "Top-5"]) {
      const cell = document.createElement("th");
      cell.textContent = label;
      head.append(cell);
    }
    table.append(head);
    for (const [name, row] of Object.entries(metrics.per_item || {})) {
      const tr = document.createElement("tr");
      for (const value of [name, String(row.n), pct(row.top1), pct(row.top5)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.append(cell);
      }
      table.append(tr);
    }
    details.append(table);

    const confusions = Object.entries(metrics.confusion || {}).flatMap(
      ([truth, predictions]) =>
        Object.entries(predictions)
          .filter(([predicted]) => predicted !== truth)
          .map(([predicted, count]) => `${truth} → ${predicted} ×${count}`),
    );
    if (confusions.length) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = `Confusions: ${confusions.join("; ")}`;
      details.append(note);
    }
    container.append(details);
  }
}

function renderMetrics(run) {
  const container = document.getElementById("eval-results");
  container.replaceChildren();
  if (!run.metrics) {
    container.textContent = "No metrics yet.";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = `
    <thead><tr>
      <th>Variant</th><th>N</th><th>Top-1</th><th>Top-1 95% CI</th>
      <th>Top-5</th><th>Macro top-1</th><th>p50 total ms</th><th>p95 total ms</th>
    </tr></thead>`;
  const body = document.createElement("tbody");
  for (const [variant, metrics] of Object.entries(run.metrics)) {
    const row = document.createElement("tr");
    const ci = metrics.top1_ci
      ? `${(metrics.top1_ci[0] * 100).toFixed(0)}–${(metrics.top1_ci[1] * 100).toFixed(0)}%`
      : "—";
    const total = metrics.latency_ms?.total || {};
    row.innerHTML = `
      <td>${variant}</td>
      <td>${metrics.n}</td>
      <td>${pct(metrics.top1)}</td>
      <td>${ci}</td>
      <td>${pct(metrics.top5)}</td>
      <td>${pct(metrics.macro_top1)}</td>
      <td>${(total.p50 || 0).toFixed(0)}</td>
      <td>${(total.p95 || 0).toFixed(0)}</td>`;
    body.append(row);
  }
  table.append(body);
  container.append(table);

  const first = Object.values(run.metrics)[0];
  const kNote = document.createElement("p");
  kNote.className = "muted";
  kNote.textContent = `Top-5 effective k: ${first?.k_effective ?? "—"} (warning when fewer than 5 items).`;
  container.append(kNote);

  if (first) {
    const dataNote = document.createElement("p");
    dataNote.className = "muted";
    dataNote.textContent =
      `Items: ${first.n_items} · ` +
      `Mask accepted: ${(first.mask_accepted_rate * 100).toFixed(0)}% · ` +
      `Near-duplicate test images: ${first.near_duplicate_count}` +
      (first.n_items < 5 ? " — insufficient data: fewer than 5 items, top-5 is not meaningful." : "");
    container.append(dataNote);
  }
  renderBreakdown(run, container);
}

function renderFailures(run) {
  const container = document.getElementById("eval-failures");
  container.replaceChildren();
  if (!run.failures?.length) {
    container.textContent = "No failures recorded.";
    return;
  }
  const heading = document.createElement("h3");
  heading.textContent = "Failures";
  container.append(heading);
  for (const failure of run.failures) {
    const card = document.createElement("div");
    card.className = "card";
    const row = document.createElement("div");
    const truth = document.createElement("strong");
    truth.textContent = failure.true_item_name;
    row.append(truth, ` → ${failure.pred_item_name || "none"} `);
    const metaSpan = document.createElement("span");
    metaSpan.className = "muted";
    metaSpan.textContent = `(${failure.variant}, ${failure.sample_status})`;
    row.append(metaSpan);
    card.append(row);
    if (failure.rendered_rel_path) {
      const link = document.createElement("a");
      link.href = `/media/${failure.rendered_rel_path}`;
      link.target = "_blank";
      link.textContent = "View frame";
      card.append(link);
    }
    container.append(card);
  }
}

async function poll(runId) {
  if (polling) clearInterval(polling);
  polling = setInterval(async () => {
    try {
      const run = await getEvaluation(runId);
      document.getElementById("evaluate-status").textContent =
        `${run.status} · ${run.progress_done}/${run.progress_total}`;
      if (run.status === "completed" || run.status === "failed") {
        clearInterval(polling);
        polling = null;
        renderMetrics(run);
        renderFailures(run);
        if (run.error) {
          document.getElementById("evaluate-status").textContent = `Failed: ${run.error}`;
        }
      }
    } catch (error) {
      clearInterval(polling);
      polling = null;
      document.getElementById("evaluate-status").textContent = error.message;
    }
  }, 1000);
}

async function loadLatest() {
  try {
    const runs = await listEvaluations();
    if (!runs.length) return;
    const run = await getEvaluation(runs[0].id);
    renderMetrics(run);
    renderFailures(run);
    document.getElementById("evaluate-status").textContent = `${run.status} · ${runs[0].id}`;
  } catch (error) {
    document.getElementById("evaluate-status").textContent = error.message;
  }
}

export function initEvaluate(metaData) {
  meta = metaData;
  renderVariants();
  document.getElementById("eval-run").addEventListener("click", async () => {
    if (!selected.size) {
      document.getElementById("evaluate-status").textContent = "Select at least one variant.";
      return;
    }
    document.getElementById("evaluate-status").textContent = "Starting…";
    try {
      const result = await startEvaluation([...selected]);
      document.getElementById("evaluate-status").textContent = `Queued ${result.run_id}`;
      poll(result.run_id);
    } catch (error) {
      document.getElementById("evaluate-status").textContent = error.message;
    }
  });
  loadLatest();
}

export function activateEvaluate() {
  // No camera needed; polling continues in the background.
}
