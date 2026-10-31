export function formatPrice(cents, symbol = "$") {
  return `${symbol}${(Number(cents) / 100).toFixed(2)}`;
}

export function parseVariantIds(text) {
  return String(text || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

export function variantParam(ids) {
  return ids.join(",");
}
