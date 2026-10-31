async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = JSON.parse(detail).detail || detail;
    } catch {
      /* keep text */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const getMeta = () => jsonFetch("/api/meta");
export const getHealth = () => jsonFetch("/api/health");
export const getItems = () => jsonFetch("/api/items");

export const createItem = (name, priceCents) =>
  jsonFetch("/api/items", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, price_cents: priceCents }),
  });

export const deleteItem = (itemId) =>
  fetch(`/api/items/${itemId}`, { method: "DELETE" });

export const getCaptures = (itemId) =>
  jsonFetch(`/api/items/${itemId}/captures?role=gallery`);

export async function uploadCaptures(itemId, sessionId, blobs) {
  const form = new FormData();
  for (const blob of blobs) form.append("files", blob, "frame.jpg");
  return jsonFetch(
    `/api/items/${itemId}/captures?role=gallery&session_id=${encodeURIComponent(sessionId)}`,
    { method: "POST", body: form },
  );
}

export const deleteCapture = (itemId, captureId) =>
  fetch(`/api/items/${itemId}/captures/${captureId}`, { method: "DELETE" });

export async function scanFrame(blob) {
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  return jsonFetch("/api/scan", { method: "POST", body: form });
}
