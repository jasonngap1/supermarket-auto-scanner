export function bestTop(variants) {
  return (
    variants
      .flatMap((variant) => variant.top_items || [])
      .sort((a, b) => b.score - a.score)[0] || null
  );
}

function maskVariants(variants) {
  return variants.filter(
    (variant) => variant.variant === "v2" || variant.variant === "v3",
  );
}

function allFailed(variants, statuses) {
  return variants.length > 0 && variants.every((v) => statuses.includes(v.status));
}

export function statusMessage(body) {
  const variants = body.variants || [];
  const masks = maskVariants(variants);
  if (!masks.length) return "OK";
  if (allFailed(masks, ["model_error"])) {
    return "Models unavailable — check /api/health.";
  }
  if (allFailed(masks, ["no_detection", "mask_rejected", "model_error"])) {
    return "Item not found — hold it in the centre of the frame.";
  }
  return "OK";
}

export function isErrorStatus(body) {
  const variants = body.variants || [];
  const masks = maskVariants(variants);
  if (!masks.length) return false;
  return allFailed(masks, ["no_detection", "mask_rejected", "model_error"]);
}
