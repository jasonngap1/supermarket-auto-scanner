export const ADD_SCORE = 0.7;
export const RELEASE_SCORE = 0.6;
export const RELEASE_FRAMES = 3;
export const SWITCH_FRAMES = 3;

export function createScanState() {
  return {
    lockedItemId: null,
    lowFrames: 0,
    switchItemId: null,
    switchFrames: 0,
  };
}

export function bestTop(body) {
  return body?.top_items?.[0] || null;
}

/**
 * Only a successful match counts as a visible item. Detection failures and
 * empty frames must reach observeScan as an absence so a locked gate can
 * release and scan the next unit.
 */
export function bestFromResponse(body) {
  return body?.status === "ok" ? bestTop(body) : null;
}

/**
 * One scan observation -> next gate state plus an item to add once.
 * A gate stays locked until the visible item disappears for RELEASE_FRAMES,
 * or a different item wins for SWITCH_FRAMES.
 */
export function observeScan(state, best) {
  if (state.lockedItemId == null) {
    if (best && best.score >= ADD_SCORE) {
      return {
        state: {
          ...createScanState(),
          lockedItemId: best.item_id,
        },
        added: best,
        released: false,
      };
    }
    return { state, added: null, released: false };
  }

  if (!best || best.score < RELEASE_SCORE) {
    const lowFrames = state.lowFrames + 1;
    if (lowFrames >= RELEASE_FRAMES) {
      return {
        state: { ...createScanState() },
        added: null,
        released: true,
      };
    }
    return {
      state: {
        ...state,
        lowFrames,
        switchItemId: null,
        switchFrames: 0,
      },
      added: null,
      released: false,
    };
  }

  if (best.item_id !== state.lockedItemId && best.score >= ADD_SCORE) {
    const sameCandidate = state.switchItemId === best.item_id;
    const switchFrames = sameCandidate ? state.switchFrames + 1 : 1;
    if (switchFrames >= SWITCH_FRAMES) {
      return {
        state: {
          ...createScanState(),
          lockedItemId: best.item_id,
        },
        added: best,
        released: false,
      };
    }
    return {
      state: {
        ...state,
        lowFrames: 0,
        switchItemId: best.item_id,
        switchFrames,
      },
      added: null,
      released: false,
    };
  }

  return {
    state: {
      ...state,
      lowFrames: 0,
      switchItemId: null,
      switchFrames: 0,
    },
    added: null,
    released: false,
  };
}
