import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

function fakeAudio() {
  const oscillator = {
    type: "",
    frequency: { setValueAtTime: vi.fn() },
    connect: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
  };
  const gain = {
    gain: {
      setValueAtTime: vi.fn(),
      exponentialRampToValueAtTime: vi.fn(),
    },
    connect: vi.fn(),
  };
  const context = {
    state: "running",
    currentTime: 1,
    destination: {},
    createOscillator: () => oscillator,
    createGain: () => gain,
    resume: vi.fn(() => Promise.resolve()),
  };
  oscillator.connect.mockReturnValue(gain);
  gain.connect.mockReturnValue(context.destination);
  return { context, oscillator, gain };
}

describe("checkout beep", () => {
  it("schedules a short tone when audio is running", async () => {
    const { context, oscillator, gain } = fakeAudio();
    vi.stubGlobal("AudioContext", function FakeAudioContext() {
      return context;
    });
    const { playBeep } = await import("./beep.js");

    playBeep();

    expect(oscillator.type).toBe("square");
    expect(oscillator.frequency.setValueAtTime).toHaveBeenCalledWith(1100, 1);
    expect(gain.gain.exponentialRampToValueAtTime).toHaveBeenCalled();
    expect(oscillator.start).toHaveBeenCalledWith(1);
    expect(oscillator.stop).toHaveBeenCalledWith(1.1);
  });

  it("does nothing when Web Audio is unavailable", async () => {
    vi.stubGlobal("AudioContext", undefined);
    vi.stubGlobal("webkitAudioContext", undefined);
    const { playBeep } = await import("./beep.js");

    expect(() => playBeep()).not.toThrow();
  });
});
