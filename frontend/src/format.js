export function formatPrice(cents, symbol = "$") {
  return `${symbol}${(Number(cents) / 100).toFixed(2)}`;
}
