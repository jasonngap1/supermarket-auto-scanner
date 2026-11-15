import { describe, expect, it } from "vitest";
import {
  ADD_SCORE,
  RELEASE_FRAMES,
  SWITCH_FRAMES,
  bestFromResponse,
  bestTop,
  createScanState,
  observeScan,
} from "./scan_logic.js";

const milk = { item_id: 1, name: "Milk", price_cents: 199, score: 0.9 };
const bread = { item_id: 2, name: "Bread", price_cents: 250, score: 0.9 };

describe("scan gate", () => {
  it("uses 0.70 as the add threshold", () => {
    expect(ADD_SCORE).toBe(0.7);
  });

  it("adds an item at the add threshold and locks", () => {
    const result = observeScan(createScanState(), milk);
    expect(result.added).toEqual(milk);
    expect(result.state.lockedItemId).toBe(milk.item_id);
    expect(milk.score).toBeGreaterThanOrEqual(ADD_SCORE);
  });

  it("does not add anything below the add threshold", () => {
    const result = observeScan(createScanState(), {
      ...milk,
      score: ADD_SCORE - 0.01,
    });
    expect(result.added).toBeNull();
    expect(result.state.lockedItemId).toBeNull();
  });

  it("does not repeat the same item while it stays visible", () => {
    let state = observeScan(createScanState(), milk).state;
    for (let index = 0; index < 10; index += 1) {
      const result = observeScan(state, milk);
      expect(result.added).toBeNull();
      state = result.state;
    }
    expect(state.lockedItemId).toBe(milk.item_id);
  });

  it("releases after a low-score absence and adds the same item again", () => {
    let state = observeScan(createScanState(), milk).state;
    let released = false;
    for (let index = 0; index < RELEASE_FRAMES; index += 1) {
      const result = observeScan(state, { ...milk, score: 0.4 });
      state = result.state;
      released = released || result.released;
    }
    expect(released).toBe(true);
    expect(state.lockedItemId).toBeNull();

    const again = observeScan(state, milk);
    expect(again.added).toEqual(milk);
  });

  it("releases when frames report no detection and re-adds the same item", () => {
    let state = observeScan(createScanState(), milk).state;
    let released = false;
    for (let index = 0; index < RELEASE_FRAMES; index += 1) {
      const result = observeScan(
        state,
        bestFromResponse({ status: "no_detection", top_items: [] }),
      );
      state = result.state;
      released = released || result.released;
    }
    expect(released).toBe(true);
    expect(state.lockedItemId).toBeNull();

    const again = observeScan(state, bestFromResponse({ status: "ok", top_items: [milk] }));
    expect(again.added).toEqual(milk);
  });

  it("ignores item lists on non-ok responses", () => {
    expect(bestFromResponse({ status: "no_detection", top_items: [milk] })).toBeNull();
    expect(bestFromResponse({ status: "model_error", top_items: [milk] })).toBeNull();
    expect(bestFromResponse({ status: "ok", top_items: [milk] })).toEqual(milk);
  });

  it("resets the absence streak when the item reappears", () => {
    let state = observeScan(createScanState(), milk).state;
    for (let index = 0; index < RELEASE_FRAMES - 1; index += 1) {
      state = observeScan(state, { ...milk, score: 0.4 }).state;
    }
    state = observeScan(state, milk).state;
    expect(state.lockedItemId).toBe(milk.item_id);
    const result = observeScan(state, { ...milk, score: 0.4 });
    expect(result.released).toBe(false);
    expect(result.state.lowFrames).toBe(1);
  });

  it("switches to a different high-confidence item after a streak", () => {
    let state = observeScan(createScanState(), milk).state;
    let added = null;
    for (let index = 0; index < SWITCH_FRAMES; index += 1) {
      const result = observeScan(state, bread);
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
