import { analyzeFrame } from "../api.js";
import { captureFrame, startCamera, stopCamera } from "../camera.js";
import { formatPrice } from "../format.js";
import { isErrorStatus, statusMessage } from "./scan_logic.js";

let meta = null;
let stream = null;
let streamRef = {};
let timer = null;
let busy = false;
let selected = new Set();

function renderVariants() {
  const container = document.getElementById("scan-variants");
  container.replaceChildren();
  selected = new Set();
  for (const variant of meta.variants) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = variant.id;
    checkbox.checked = variant.id !== "v3";
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(variant.id);
      else selected.delete(variant.id);
    });
    if (checkbox.checked) selected.add(variant.id);
    label.append(checkbox, ` ${variant.label}`);
    container.append(label);
  }
}

function renderResult(body) {
  document.getElementById("scan-annotated").src =
    `data:image/jpeg;base64,${body.annotated_jpeg_b64}`;
  const status = document.getElementById("scan-status");
  status.textContent = statusMessage(body);
  status.classList.toggle("error", isErrorStatus(body));

  const results = document.getElementById("scan-results");
  results.replaceChildren();
  for (const variant of body.variants) {
    const card = document.createElement("div");
    card.className = "card";
    const title = document.createElement("div");
    title.innerHTML = `<strong>${
      meta.variants.find((v) => v.id === variant.variant)?.label || variant.variant
    }</strong> <span class="muted">${variant.status}</span>`;
    card.append(title);
    if (variant.status === "ok" && variant.top_items.length) {
      for (const [index, item] of variant.top_items.entries()) {
        const row = document.createElement("div");
        row.textContent = `${index + 1}. ${item.name} — ${formatPrice(
          item.price_cents,
          meta.currency_symbol,
        )} (${item.score.toFixed(3)})`;
        card.append(row);
      }
    } else {
      const row = document.createElement("div");
      row.className = "muted";
      row.textContent =
        variant.status === "no_gallery"
          ? "No gallery captures yet — enroll items first."
          : variant.status;
      card.append(row);
    }
    results.append(card);
  }

  document.getElementById("scan-timings").textContent = JSON.stringify(
    body.variants.map((v) => ({ variant: v.variant, timings_ms: v.timings_ms })),
    null,
    2,
  );
}

async function analyzeOnce() {
  if (busy) return;
  if (!selected.size) {
    document.getElementById("scan-status").textContent = "Select at least one variant.";
    return;
  }
  const video = document.getElementById("scan-video");
  if (!video.videoWidth) return;
  busy = true;
  const status = document.getElementById("scan-status");
  status.textContent = "Analyzing…";
  status.classList.remove("error");
  try {
    const blob = await captureFrame(video);
    const body = await analyzeFrame(blob, [...selected]);
    renderResult(body);
  } catch (error) {
    status.textContent = error.message;
    status.classList.add("error");
  } finally {
    busy = false;
  }
}

function setAuto(enabled) {
  if (timer) clearInterval(timer);
  timer = enabled ? setInterval(analyzeOnce, 1100) : null;
}

export function initScan(metaData) {
  meta = metaData;
  renderVariants();
  document.getElementById("scan-analyze").addEventListener("click", analyzeOnce);
  document.getElementById("scan-auto").addEventListener("change", (event) => {
    setAuto(event.target.checked);
  });
}

export async function activateScan() {
  if (!stream) {
    stream = await startCamera(document.getElementById("scan-video"), streamRef);
  }
}

export function deactivateScan() {
  document.getElementById("scan-auto").checked = false;
  setAuto(false);
  stopCamera(stream);
  stream = null;
  streamRef = {};
}
