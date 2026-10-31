import { formatPrice } from "../format.js";
import { cartQuantity, cartTotalCents, lineTotalCents } from "./cart_logic.js";

export function renderCart(container, lines, meta, { onRemove }) {
  container.replaceChildren();
  if (!lines.length) {
    const empty = document.createElement("p");
    empty.className = "cart-empty";
    empty.textContent = "Show an item to the camera.";
    container.append(empty);
    return;
  }

  for (const line of lines) {
    const row = document.createElement("div");
    row.className = "cart-line";

    const imageWrap = document.createElement("div");
    imageWrap.className = "cart-thumb";
    if (line.thumb_rel_path) {
      const image = document.createElement("img");
      image.src = `/media/${line.thumb_rel_path}`;
      image.alt = line.name;
      imageWrap.append(image);
    } else {
      imageWrap.textContent = "—";
    }

    const details = document.createElement("div");
    details.className = "cart-line-details";
    const name = document.createElement("strong");
    name.textContent = line.name;
    const unit = document.createElement("span");
    unit.className = "muted";
    unit.textContent = `${formatPrice(line.price_cents, meta.currency_symbol)} each`;
    details.append(name, unit);

    const qty = document.createElement("span");
    qty.className = "cart-qty";
    qty.textContent = `×${line.qty}`;

    const total = document.createElement("span");
    total.className = "cart-line-total";
    total.textContent = formatPrice(lineTotalCents(line), meta.currency_symbol);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "cart-remove";
    remove.title = "Remove item";
    remove.setAttribute("aria-label", `Remove ${line.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => onRemove(line.item_id));

    row.append(imageWrap, details, qty, total, remove);
    container.append(row);
  }
}

export function renderCartTotal(container, lines, meta, countContainer = null) {
  container.textContent = formatPrice(cartTotalCents(lines), meta.currency_symbol);
  if (countContainer) {
    const count = cartQuantity(lines);
    countContainer.textContent = `${count} item${count === 1 ? "" : "s"}`;
  }
}
