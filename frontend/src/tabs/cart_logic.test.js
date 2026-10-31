import { describe, expect, it } from "vitest";
import {
  addToCart,
  cartQuantity,
  cartTotalCents,
  lineTotalCents,
  removeFromCart,
} from "./cart_logic.js";

const milk = {
  item_id: 1,
  name: "Milk",
  price_cents: 199,
  thumb_rel_path: "thumbs/milk.jpg",
};
const bread = { item_id: 2, name: "Bread", price_cents: 250, thumb_rel_path: null };

describe("cart", () => {
  it("adds a new line with quantity one", () => {
    const lines = addToCart([], milk);
    expect(lines).toHaveLength(1);
    expect(lines[0].qty).toBe(1);
    expect(lines[0].thumb_rel_path).toBe("thumbs/milk.jpg");
  });

  it("increments quantity for an existing item", () => {
    const lines = addToCart(addToCart([], milk), milk);
    expect(lines).toHaveLength(1);
    expect(lines[0].qty).toBe(2);
  });

  it("keeps separate lines for different items", () => {
    const lines = addToCart(addToCart([], milk), bread);
    expect(lines).toHaveLength(2);
    expect(cartQuantity(lines)).toBe(2);
  });

  it("computes line and order totals in cents", () => {
    const lines = addToCart(addToCart(addToCart([], milk), milk), bread);
    expect(lineTotalCents(lines[0])).toBe(398);
    expect(cartTotalCents(lines)).toBe(648);
  });

  it("removes a line", () => {
    const lines = addToCart(addToCart([], milk), bread);
    expect(removeFromCart(lines, milk.item_id)).toEqual([lines[1]]);
  });
});
