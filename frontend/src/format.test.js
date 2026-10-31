import { describe, expect, it } from "vitest";
import { formatPrice } from "./format.js";

describe("format helpers", () => {
  it("formats integer cents", () => {
    expect(formatPrice(199, "$")).toBe("$1.99");
    expect(formatPrice(0, "S$")).toBe("S$0.00");
  });
});
