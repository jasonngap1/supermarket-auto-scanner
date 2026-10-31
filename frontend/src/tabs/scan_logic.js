export function bestTop(variants) {
  return (
    variants
      .flatMap((variant) => variant.top_items || [])
      .sort((a, b) => b.score - a.score)[0] || null
  );
}

function maskVariants(variants) {
  return variants.filter((variant) => ["v2", "v3"].includes(variant.variant));
}

function handVariants(variants) {
  return variants.filter((variant) => variant.variant === "v1");
}

function allFailed(variants, statuses) {
  return variants.length > 0 && variants.every((v) => statuses.includes(v.status));
}

export function statusMessage(body) {
  const variants = body.variants || [];
  const masks = maskVariants(variants);
  if (masks.length) {
    if (allFailed(masks, ["model_error"])) return "Models unavailable — check /api/health.";
    if (allFailed(masks, ["mask_rejected", "model_error"])) {
      return "Item not found — hold it in the centre of the frame.";
    }
    return "OK";
  }
  const hands = handVariants(variants);
  if (hands.length) {
    if (allFailed(hands, ["no_hand", "model_error"])) {
      return "No hand detected — hold the item up to the camera.";
    }
  }
  return "OK";
}

export function isErrorStatus(body) {
  const variants = body.variants || [];
  const masks = maskVariants(variants);
  if (masks.length) {
    return allFailed(masks, ["mask_rejected", "model_error"]);
  }
  const hands = handVariants(variants);
  if (hands.length) {
    return allFailed(hands, ["no_hand", "model_error"]);
  }
  return false;
}
