import { scanFrame } from "../api.js";
import { captureFrame, startCamera, stopCamera } from "../camera.js";
import { renderCart, renderCartTotal } from "./cart.js";
import { addToCart, removeFromCart } from "./cart_logic.js";
import {
  bestTop,
  createScanState,
  observeScan,
} from "./scan_logic.js";

const SCAN_INTERVAL_MS = 1100;

let meta = null;
let stream = null;
let streamRef = {};
let timer = null;
let busy = false;
let active = false;
let scanState = createScanState();
let cartLines = [];

function status(text, error = false) {
  const element = document.getElementById("scan-status");
  element.textContent = text;
  element.classList.toggle("error", error);
}

function renderCartPanel() {
  renderCart(document.getElementById("cart-lines"), cartLines, meta, {
    onRemove: (itemId) => {
      cartLines = removeFromCart(cartLines, itemId);
      renderCartPanel();
      status("Item removed. Show the next item.");
    },
  });
  renderCartTotal(
    document.getElementById("cart-total"),
    cartLines,
    meta,
    document.getElementById("cart-count"),
  );
}

function clearCart() {
  cartLines = [];
  scanState = createScanState();
  renderCartPanel();
  status("Order cleared. Show an item.");
}

function applyScan(body) {
  if (body.status === "no_gallery") {
    status("Add items with a gallery photo first.");
    return;
  }

  const best = bestTop(body);
  const result = observeScan(scanState, best);
  scanState = result.state;

  if (result.added) {
    cartLines = addToCart(cartLines, result.added);
    renderCartPanel();
    status(`Added ${result.added.name}. Remove it to scan the next item.`);
    return;
  }

  if (result.released) {
    status("Item left. Show the next item.");
    return;
  }

  if (scanState.lockedItemId != null) {
    status("Item added. Remove it to scan the next item.");
    return;
  }

  status("Show an item to the camera.");
}

async function analyzeOnce() {
  if (!active || busy) return;
  const video = document.getElementById("scan-video");
  if (!video.videoWidth) return;
  busy = true;
  try {
    const blob = await captureFrame(video);
    applyScan(await scanFrame(blob));
  } catch (error) {
    status(error.message, true);
  } finally {
    busy = false;
  }
}

export function initScan(metaData) {
  meta = metaData;
  renderCartPanel();
  document.getElementById("cart-clear").addEventListener("click", clearCart);
}

export async function activateScan() {
  active = true;
  if (!stream) {
    stream = await startCamera(document.getElementById("scan-video"), streamRef);
  }
  if (!timer) timer = setInterval(analyzeOnce, SCAN_INTERVAL_MS);
  analyzeOnce();
}

export function deactivateScan() {
  active = false;
  if (timer) clearInterval(timer);
  timer = null;
  stopCamera(stream);
  stream = null;
  streamRef = {};
}
