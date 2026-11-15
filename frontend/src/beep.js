let context = null;

function audioContext() {
  const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctor) return null;
  if (!context) context = new Ctor();
  return context;
}

function startTone(ctx) {
  const now = ctx.currentTime;
  const oscillator = ctx.createOscillator();
  const gain = ctx.createGain();
  oscillator.type = "square";
  oscillator.frequency.setValueAtTime(1100, now);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.08, now + 0.006);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);
  oscillator.connect(gain).connect(ctx.destination);
  oscillator.start(now);
  oscillator.stop(now + 0.1);
}

/** Unlock audio after the first user gesture; browsers block autoplay. */
export function primeBeep() {
  const ctx = audioContext();
  if (!ctx) return;
  const unlock = () => {
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    if (ctx.state !== "running" || typeof document === "undefined") return;
    document.removeEventListener("pointerdown", unlock);
    document.removeEventListener("keydown", unlock);
  };
  unlock();
  if (typeof document !== "undefined" && ctx.state !== "running") {
    document.addEventListener("pointerdown", unlock);
    document.addEventListener("keydown", unlock);
  }
}

/** Short checkout-style beep for a successful cart add. */
export function playBeep() {
  const ctx = audioContext();
  if (!ctx) return;
  if (ctx.state === "running") {
    startTone(ctx);
    return;
  }
  ctx.resume().then(() => startTone(ctx)).catch(() => {});
}
