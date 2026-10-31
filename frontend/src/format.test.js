import { describe, expect, it } from "vitest";
import { formatPrice, parseVariantIds, variantParam } from "./format.js";

describe("format helpers", () => {
  it("formats integer cents", () => {
    expect(formatPrice(199, "$")).toBe("$1.99");
    expect(formatPrice(0, "S$")).toBe("S$0.00");
  });

  it("parses and serializes variant ids", () => {
    expect(parseVariantIds(" v0, v2 ,,")).toEqual(["v0", "v2"]);
    expect(variantParam(["v0", "v2"])).toBe("v0,v2");
  });
});
