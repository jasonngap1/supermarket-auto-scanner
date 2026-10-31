import { describe, expect, it } from "vitest";
import { bestTop, isErrorStatus, statusMessage } from "./scan_logic.js";

describe("scan logic", () => {
  it("picks the highest-scoring top-1 across variants", () => {
    const best = bestTop([
      { top_items: [{ name: "Milk", score: 0.7 }] },
      { top_items: [{ name: "Bread", score: 0.9 }] },
    ]);
    expect(best.name).toBe("Bread");
  });

  it("asks for a centred item when the mask is missing", () => {
    expect(
      statusMessage({
        hand_found: false,
        mask_found: false,
        variants: [{ variant: "v2", status: "mask_rejected" }],
      }),
    ).toMatch(/centre/);
    expect(
      statusMessage({
        hand_found: true,
        mask_found: true,
        variants: [{ variant: "v2", status: "ok" }],
      }),
    ).toBe("OK");
  });

  it("does not demand a hand for mask variants", () => {
    const body = {
      hand_found: false,
      mask_found: true,
      variants: [
        { variant: "v0", status: "ok" },
        { variant: "v2", status: "ok" },
      ],
    };
    expect(statusMessage(body)).toBe("OK");
    expect(isErrorStatus(body)).toBe(false);
  });

  it("does not demand a mask when only V0 is selected", () => {
    const body = {
      hand_found: false,
      mask_found: false,
      variants: [{ variant: "v0", status: "ok" }],
    };
    expect(statusMessage(body)).toBe("OK");
    expect(isErrorStatus(body)).toBe(false);
  });

  it("flags a no-hand result when the hand-crop variant is selected", () => {
    const body = {
      hand_found: false,
      mask_found: true,
      variants: [
        { variant: "v0", status: "ok" },
        { variant: "v1", status: "no_hand" },
      ],
    };
    expect(statusMessage(body)).toMatch(/No hand/);
    expect(isErrorStatus(body)).toBe(true);
  });

  it("flags a rejected mask as an error", () => {
    const body = {
      hand_found: true,
      mask_found: false,
      variants: [{ variant: "v2", status: "mask_rejected" }],
    };
    expect(isErrorStatus(body)).toBe(true);
  });

  it("ignores the hand control failing when the mask pipeline works", () => {
    const body = {
      hand_found: false,
      mask_found: true,
      variants: [
        { variant: "v0", status: "ok" },
        { variant: "v1", status: "no_hand" },
        { variant: "v2", status: "ok" },
      ],
    };
    expect(statusMessage(body)).toBe("OK");
    expect(isErrorStatus(body)).toBe(false);
  });
});
