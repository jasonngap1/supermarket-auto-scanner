export function addToCart(lines, item) {
  const existing = lines.find((line) => line.item_id === item.item_id);
  if (existing) {
    return lines.map((line) =>
      line.item_id === item.item_id ? { ...line, qty: line.qty + 1 } : line,
    );
  }
  return [
    ...lines,
    {
      item_id: item.item_id,
      name: item.name,
      price_cents: item.price_cents,
      thumb_rel_path: item.thumb_rel_path || null,
      qty: 1,
    },
  ];
}

export function removeFromCart(lines, itemId) {
  return lines.filter((line) => line.item_id !== itemId);
}

export function lineTotalCents(line) {
  return line.price_cents * line.qty;
}

export function cartTotalCents(lines) {
  return lines.reduce((total, line) => total + lineTotalCents(line), 0);
}

export function cartQuantity(lines) {
  return lines.reduce((total, line) => total + line.qty, 0);
}
