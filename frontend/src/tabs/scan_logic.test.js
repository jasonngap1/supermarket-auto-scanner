import { describe, expect, it } from "vitest";
import {
  ADD_SCORE,
  RELEASE_MS,
  SAME_VIEW_DISTANCE,
  SWITCH_FRAMES,
  bestFromResponse,
  bestTop,
  createScanState,
  observeScan,
  signatureDistance,
} from "./scan_logic.js";

const milk = { item_id: 1, name: "Milk", price_cents: 199, score: 0.9 };
const bread = { item_id: 2, name: "Bread", price_cents: 250, score: 0.9 };

// Two clearly different 8x8 luma views: the held item vs. the bare shelf.
const itemView = Array.from({ length: 64 }, (_, index) =>
  index % 8 < 4 ? 200 : 40,
);
const emptyView = Array.from({ length: 64 }, (_, index) =>
  index % 8 < 4 ? 40 : 200,
);
const brighterItemView = itemView.map((value) => value + 10);

const frame = (now, overrides = {}) => ({
  now,
  best: null,
  maskRejected: false,
  signature: itemView,
  ...overrides,
});

describe("scan gate", () => {
  it("uses 0.70 as the add threshold", () => {
    expect(ADD_SCORE).toBe(0.7);
  });

  it("adds an item at the add threshold and locks", () => {
    const result = observeScan(createScanState(), frame(0, { best: milk }));
    expect(result.added).toEqual(milk);
    expect(result.state.lockedItemId).toBe(milk.item_id);
    expect(milk.score).toBeGreaterThanOrEqual(ADD_SCORE);
  });

  it("does not add anything below the add threshold", () => {
    const result = observeScan(
      createScanState(),
      frame(0, { best: { ...milk, score: ADD_SCORE - 0.01 } }),
    );
    expect(result.added).toBeNull();
    expect(result.state.lockedItemId).toBeNull();
  });

  it("does not repeat the same item while it stays visible", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    for (let index = 0; index < 10; index += 1) {
      const result = observeScan(state, frame(index * 300, { best: milk }));
      expect(result.added).toBeNull();
      state = result.state;
    }
    expect(state.lockedItemId).toBe(milk.item_id);
  });

  it("keeps the lock through a long detector dropout over the same view", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    let released = false;
    for (let index = 1; index <= 12; index += 1) {
      const result = observeScan(state, frame(index * 300));
      expect(result.added).toBeNull();
      released = released || result.released;
      state = result.state;
    }
    expect(released).toBe(false);
    expect(state.lockedItemId).toBe(milk.item_id);

    const again = observeScan(state, frame(4000, { best: milk }));
    expect(again.added).toBeNull();
    expect(again.state.lockedItemId).toBe(milk.item_id);
  });

  it("does not count a rejected mask as an absent item", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    for (let index = 1; index <= 6; index += 1) {
      const result = observeScan(
        state,
        frame(index * 300, { maskRejected: true, signature: emptyView }),
      );
      expect(result.released).toBe(false);
      state = result.state;
    }
    expect(state.lockedItemId).toBe(milk.item_id);
    expect(state.absentSince).toBeNull();
  });

  it("releases only after the absence window over a changed view", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    state = observeScan(state, frame(100, { signature: emptyView })).state;
    state = observeScan(
      state,
      frame(100 + RELEASE_MS - 100, { signature: emptyView }),
    ).state;
    expect(state.lockedItemId).toBe(milk.item_id);

    const result = observeScan(
      state,
      frame(100 + RELEASE_MS, { signature: emptyView }),
    );
    expect(result.released).toBe(true);
    expect(result.state.lockedItemId).toBeNull();

    const again = observeScan(
      result.state,
      frame(100 + RELEASE_MS + 300, { best: milk }),
    );
    expect(again.added).toEqual(milk);
  });

  it("releases on a changed view even without signatures", () => {
    let state = observeScan(
      createScanState(),
      frame(0, { best: milk, signature: null }),
    ).state;
    let released = false;
    for (const now of [400, 800, 400 + RELEASE_MS]) {
      const result = observeScan(state, frame(now, { signature: null }));
      released = released || result.released;
      state = result.state;
    }
    expect(released).toBe(true);
    expect(state.lockedItemId).toBeNull();
  });

  it("treats a global brightness shift as the same view", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    for (let index = 1; index <= 8; index += 1) {
      const result = observeScan(
        state,
        frame(index * 300, { signature: brighterItemView }),
      );
      expect(result.released).toBe(false);
      state = result.state;
    }
    expect(state.lockedItemId).toBe(milk.item_id);
  });

  it("releases when frames report no detection and re-adds the same item", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    let released = false;
    for (const now of [400, 800, 400 + RELEASE_MS]) {
      const result = observeScan(
        state,
        frame(now, { signature: emptyView }),
      );
      state = result.state;
      released = released || result.released;
    }
    expect(released).toBe(true);
    expect(state.lockedItemId).toBeNull();

    const again = observeScan(
      state,
      frame(400 + RELEASE_MS + 300, { best: milk }),
    );
    expect(again.added).toEqual(milk);
  });

  it("ignores item lists on non-ok responses", () => {
    expect(bestFromResponse({ status: "no_detection", top_items: [milk] })).toBeNull();
    expect(bestFromResponse({ status: "model_error", top_items: [milk] })).toBeNull();
    expect(bestFromResponse({ status: "ok", top_items: [milk] })).toEqual(milk);
  });

  it("resets the absence streak when the item reappears", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    state = observeScan(state, frame(400, { signature: emptyView })).state;
    state = observeScan(state, frame(800, { signature: emptyView })).state;
    state = observeScan(state, frame(1000, { best: milk })).state;
    expect(state.lockedItemId).toBe(milk.item_id);
    expect(state.absentSince).toBeNull();

    const result = observeScan(state, frame(1300, { signature: emptyView }));
    expect(result.released).toBe(false);
    expect(result.state.absentSince).toBe(1300);
  });

  it("switches to a different high-confidence item after a streak", () => {
    let state = observeScan(createScanState(), frame(0, { best: milk })).state;
    let added = null;
    for (let index = 1; index <= SWITCH_FRAMES; index += 1) {
      const result = observeScan(
        state,
        frame(index * 300, { best: bread, signature: emptyView }),
      );
      state = result.state;
      added = added || result.added;
    }
    expect(added).toEqual(bread);
    expect(state.lockedItemId).toBe(bread.item_id);
  });

  it("takes top-1 from the scan response", () => {
    expect(bestTop({ top_items: [milk, bread] })).toEqual(milk);
    expect(bestTop({ top_items: [] })).toBeNull();
  });
});

describe("signatureDistance", () => {
  it("reports zero for the same view", () => {
    expect(signatureDistance(itemView, itemView)).toBe(0);
  });

  it("ignores a global brightness shift", () => {
    expect(signatureDistance(itemView, brighterItemView)).toBe(0);
  });

  it("reports a large distance for a changed scene", () => {
    expect(signatureDistance(itemView, emptyView)).toBeGreaterThan(
      SAME_VIEW_DISTANCE,
    );
  });

  it("treats missing or mismatched signatures as no evidence", () => {
    expect(signatureDistance(null, itemView)).toBe(Infinity);
    expect(signatureDistance(itemView, emptyView.slice(0, 32))).toBe(Infinity);
  });
});
