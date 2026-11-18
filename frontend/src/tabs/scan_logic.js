export const ADD_SCORE = 0.7;
export const RELEASE_SCORE = 0.7;
export const CONFIRM_MS = 500;
export const RELEASE_MS = 1200;
export const SWITCH_FRAMES = 3;
// Mean-centred mean absolute difference between two 8x8 luma signatures.
// Below this the camera view still holds the locked presentation, so a missed
// detection is a model dropout, not an empty view.
export const SAME_VIEW_DISTANCE = 10;

export function createScanState() {
  return {
    lockedItemId: null,
    referenceSignature: null,
    absentSince: null,
    switchItemId: null,
    switchFrames: 0,
    pendingItemId: null,
    pendingBest: null,
    pendingSignature: null,
    pendingSince: null,
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
 * Mean absolute difference of two luma signatures after removing each one's
 * mean, so a global brightness or exposure shift does not read as a new scene.
 * `null` signatures (canvas unavailable) report Infinity: no continuity
 * evidence, fall back to the time-based release.
 */
export function signatureDistance(first, second) {
  if (!first || !second || first.length === 0 || first.length !== second.length) {
    return Infinity;
  }
  const meanFirst = first.reduce((sum, value) => sum + value, 0) / first.length;
  const meanSecond = second.reduce((sum, value) => sum + value, 0) / second.length;
  let total = 0;
  for (let index = 0; index < first.length; index += 1) {
    const firstValue = first[index] - meanFirst;
    const secondValue = second[index] - meanSecond;
    total += Math.abs(firstValue - secondValue);
  }
  return total / first.length;
}

/**
 * One scan observation -> next gate state plus an item to add once.
 *
 * observation = { now, best, maskRejected, signature }
 * - now: monotonic ms timestamp of this frame.
 * - best: top match, or null when the frame did not match.
 * - maskRejected: the backend found a detection but rejected its mask, so the
 *   held item did not leave the view.
 * - signature: 8x8 luma signature of the captured frame, or null.
 *
 * A match must hold for CONFIRM_MS before it is added, so the item is settled
 * in front of the camera. A gate stays locked until the item is absent for
 * RELEASE_MS *and* the view has changed from the locked presentation, or a
 * different item wins for SWITCH_FRAMES. A model dropout over an unchanged
 * scene never releases it, which is what stops one physical item from being
 * added twice.
 */
export function observeScan(state, observation = {}) {
  const {
    now = 0,
    best = null,
    maskRejected = false,
    signature = null,
  } = observation;

  if (state.lockedItemId == null) {
    if (best && best.score >= ADD_SCORE) {
      const sameCandidate = state.pendingItemId === best.item_id;
      const pendingSince = sameCandidate ? (state.pendingSince ?? now) : now;
      if (now - pendingSince >= CONFIRM_MS) {
        return {
          state: {
            ...createScanState(),
            lockedItemId: best.item_id,
            referenceSignature: signature ?? state.pendingSignature,
          },
          added: best,
          released: false,
        };
      }
      return {
        state: {
          ...state,
          pendingItemId: best.item_id,
          pendingBest: best,
          pendingSignature: signature ?? state.pendingSignature,
          pendingSince,
        },
        added: null,
        released: false,
      };
    }

    // Without a current match, a pending candidate survives only while the
    // camera still shows it. The add waits for the next positive frame so the
    // item is matched on evidence from the settled presentation.
    if (state.pendingItemId != null) {
      const viewHeld =
        signature != null &&
        state.pendingSignature != null &&
        signatureDistance(signature, state.pendingSignature) <=
          SAME_VIEW_DISTANCE;
      if (!maskRejected && !viewHeld) {
        return {
          state: createScanState(),
          added: null,
          released: false,
        };
      }
    }
    return { state, added: null, released: false };
  }

  const stillMatched =
    best != null &&
    best.item_id === state.lockedItemId &&
    best.score >= RELEASE_SCORE;
  if (stillMatched) {
    return {
      state: {
        ...state,
        referenceSignature: signature ?? state.referenceSignature,
        absentSince: null,
        switchItemId: null,
        switchFrames: 0,
      },
      added: null,
      released: false,
    };
  }

  if (best && best.item_id !== state.lockedItemId && best.score >= ADD_SCORE) {
    const sameCandidate = state.switchItemId === best.item_id;
    const switchFrames = sameCandidate ? state.switchFrames + 1 : 1;
    if (switchFrames >= SWITCH_FRAMES) {
      return {
        state: {
          ...createScanState(),
          lockedItemId: best.item_id,
          referenceSignature: signature,
        },
        added: best,
        released: false,
      };
    }
    return {
      state: {
        ...state,
        absentSince: null,
        switchItemId: best.item_id,
        switchFrames,
      },
      added: null,
      released: false,
    };
  }

  // No match. Absence counts only when the item is not localized and the
  // camera no longer shows the locked presentation.
  const viewHeld =
    signature != null &&
    state.referenceSignature != null &&
    signatureDistance(signature, state.referenceSignature) <= SAME_VIEW_DISTANCE;
  if (maskRejected || viewHeld) {
    return {
      state: {
        ...state,
        absentSince: null,
        switchItemId: null,
        switchFrames: 0,
      },
      added: null,
      released: false,
    };
  }

  const absentSince = state.absentSince ?? now;
  if (now - absentSince >= RELEASE_MS) {
    return {
      state: { ...createScanState() },
      added: null,
      released: true,
    };
  }
  return {
    state: {
      ...state,
      absentSince,
      switchItemId: null,
      switchFrames: 0,
    },
    added: null,
    released: false,
  };
}
