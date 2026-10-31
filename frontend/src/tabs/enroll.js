import {
  createItem,
  deleteCapture,
  deleteItem,
  getCaptures,
  getItems,
  uploadCaptures,
} from "../api.js";
import { captureFrame, startCamera, stopCamera } from "../camera.js";
import { formatPrice } from "../format.js";

let meta = null;
let selectedItem = null;
let stream = null;
let streamRef = {};
let sessionId = crypto.randomUUID().slice(0, 8);

function itemButton(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = `${item.name} — ${formatPrice(
    item.price_cents,
    meta.currency_symbol,
  )} (gallery ${item.gallery_count} / test ${item.test_count})`;
  button.addEventListener("click", () => selectItem(item));
  return button;
}

async function refreshItems() {
  const items = await getItems();
  const list = document.getElementById("item-list");
  list.replaceChildren();
  for (const item of items) list.append(itemButton(item));
  if (selectedItem) {
    const updated = items.find((item) => item.id === selectedItem.id);
    if (!updated) clearSelection();
    else {
      selectedItem = updated;
      document.getElementById("selected-item-name").textContent = updated.name;
    }
  }
}

function clearSelection() {
  selectedItem = null;
  document.getElementById("enroll-panel").hidden = true;
  document.getElementById("capture-grid").replaceChildren();
  deactivateEnroll();
}

async function selectItem(item) {
  selectedItem = item;
  document.getElementById("enroll-panel").hidden = false;
  document.getElementById("selected-item-name").textContent = item.name;
  document.getElementById("enroll-session").value = sessionId;
  await activateEnroll();
  await refreshCaptures();
}

async function refreshCaptures() {
  if (!selectedItem) return;
  const [gallery, test] = await Promise.all([
    getCaptures(selectedItem.id, "gallery"),
    getCaptures(selectedItem.id, "test"),
  ]);
  document.getElementById("capture-counts").textContent =
    `Gallery: ${gallery.length} · Test: ${test.length}`;
  const grid = document.getElementById("capture-grid");
  grid.replaceChildren();
  for (const capture of [...gallery, ...test]) {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = `/media/${capture.thumb_rel_path}`;
    image.alt = `${capture.role} capture`;
    const caption = document.createElement("figcaption");
    caption.textContent = `${capture.role} · ${capture.mask_status}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Delete";
    remove.className = "danger";
    remove.addEventListener("click", async () => {
      await deleteCapture(selectedItem.id, capture.id);
      await refreshCaptures();
    });
    figure.append(image, caption, remove);
    grid.append(figure);
  }
}

async function capture(role, button) {
  if (!selectedItem) return;
  button.disabled = true;
  const status = document.getElementById("enroll-status");
  status.textContent = "Capturing and segmenting…";
  try {
    const blob = await captureFrame(document.getElementById("enroll-video"));
    const result = await uploadCaptures(
      selectedItem.id,
      role,
      document.getElementById("enroll-session").value || sessionId,
      [blob],
    );
    if (result.accepted.length) {
      const accepted = result.accepted[0];
      status.textContent = `Accepted ${role} capture.`;
      if (accepted.near_duplicate) {
        status.textContent +=
          ` ⚠ Near-duplicate of an existing ${role === "test" ? "gallery" : "test"} image` +
          ` (${accepted.near_duplicate_score.toFixed(3)}); use a different angle/session.`;
      }
    } else {
      const reason = result.rejected[0]?.reason || "unknown";
      const message =
        {
          mask_rejected: "item not found — hold it in the centre of the frame",
          no_hand: "no hand detected — hold the item up to the camera",
          duplicate: "duplicate image",
          model_error: "model unavailable",
        }[reason] || reason;
      status.textContent = `Rejected: ${message} — retake.`;
    }
    await refreshItems();
    await refreshCaptures();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

export function initEnroll(metaData) {
  meta = metaData;
  document.getElementById("item-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = document.getElementById("item-name").value.trim();
    const price = Number(document.getElementById("item-price").value);
    if (!name || Number.isNaN(price) || price < 0) return;
    await createItem(name, Math.round(price * 100));
    event.target.reset();
    await refreshItems();
  });

  document.getElementById("enroll-new-session").addEventListener("click", () => {
    sessionId = crypto.randomUUID().slice(0, 8);
    document.getElementById("enroll-session").value = sessionId;
  });

  document.querySelectorAll("#enroll-panel [data-role]").forEach((button) => {
    button.addEventListener("click", () => capture(button.dataset.role, button));
  });

  document.getElementById("enroll-delete-item").addEventListener("click", async () => {
    if (!selectedItem) return;
    if (!confirm(`Delete ${selectedItem.name} and all its captures?`)) return;
    await deleteItem(selectedItem.id);
    clearSelection();
    await refreshItems();
  });

  refreshItems().catch((error) => {
    document.getElementById("enroll-status").textContent = error.message;
  });
}

export async function activateEnroll() {
  if (!selectedItem) return;
  if (!stream) {
    stream = await startCamera(document.getElementById("enroll-video"), streamRef);
  }
}

export function deactivateEnroll() {
  stopCamera(stream);
  stream = null;
  streamRef = {};
}
