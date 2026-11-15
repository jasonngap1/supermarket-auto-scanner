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

  it("asks for a centred item when detection or the mask fails", () => {
    expect(
      statusMessage({ variants: [{ variant: "v2", status: "mask_rejected" }] }),
    ).toMatch(/centre/);
    expect(
      statusMessage({ variants: [{ variant: "v2", status: "no_detection" }] }),
    ).toMatch(/centre/);
    expect(
      statusMessage({ variants: [{ variant: "v2", status: "ok" }] }),
    ).toBe("OK");
  });

  it("does not demand a mask when only V0 is selected", () => {
    const body = { variants: [{ variant: "v0", status: "ok" }] };
    expect(statusMessage(body)).toBe("OK");
    expect(isErrorStatus(body)).toBe(false);
  });

  it("flags a model failure", () => {
    const body = { variants: [{ variant: "v2", status: "model_error" }] };
    expect(statusMessage(body)).toMatch(/Models unavailable/);
    expect(isErrorStatus(body)).toBe(true);
  });

  it("treats an accepted mask as success", () => {
    const body = {
      variants: [
        { variant: "v0", status: "ok" },
        { variant: "v2", status: "ok" },
      ],
    };
    expect(statusMessage(body)).toBe("OK");
    expect(isErrorStatus(body)).toBe(false);
  });
});
