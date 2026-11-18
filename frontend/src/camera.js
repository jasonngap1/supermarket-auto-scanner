export async function startCamera(video, streamRef = {}) {
  if (streamRef.stream) return streamRef.stream;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  streamRef.stream = stream;
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopCamera(stream) {
  if (stream) stream.getTracks().forEach((track) => track.stop());
}

/** 8x8 luma thumbnail of the current view, used to tell a model dropout from
 * an actually empty view. 64 values, 0..255. */
export function frameSignature(video, size = 8) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(video, 0, 0, size, size);
  const { data } = context.getImageData(0, 0, size, size);
  const values = new Array(size * size);
  for (let index = 0; index < values.length; index += 1) {
    const offset = index * 4;
    values[index] =
      0.299 * data[offset] + 0.587 * data[offset + 1] + 0.114 * data[offset + 2];
  }
  return values;
}

export function captureFrame(video, quality = 0.9) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode frame"))),
      "image/jpeg",
      quality,
    );
  });
}
