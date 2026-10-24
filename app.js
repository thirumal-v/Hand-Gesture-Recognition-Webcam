import {
  FilesetResolver,
  GestureRecognizer,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
const MIN_CONFIDENCE = 0.58;
const STABLE_FRAMES = 3;

const gestureCatalog = [
  { id: "Closed_Fist", pattern: "00000", label: "Fist", emoji: "✊", description: "All fingers curled" },
  { id: "Pinky_Up", pattern: "00001", label: "Pinky", emoji: "🤙", description: "Pinky finger up" },
  { id: "Ring_Up", pattern: "00010", label: "Ring", emoji: "☝️", description: "Ring finger up" },
  { id: "Ring_Pinky", pattern: "00011", label: "Ring + pinky", emoji: "🤘", description: "Ring and pinky up" },
  { id: "Middle_Ring", pattern: "00110", label: "Middle + ring", emoji: "✌️", description: "Middle and ring up" },
  { id: "Middle_Ring_Pinky", pattern: "00111", label: "Middle, ring + pinky", emoji: "🖐️", description: "Three fingers up" },
  { id: "Pointing_Up", pattern: "01000", label: "One", emoji: "☝️", description: "Index finger up" },
  { id: "Index_Pinky", pattern: "01001", label: "Index + pinky", emoji: "🤘", description: "Index and pinky up" },
  { id: "Index_Ring_Pinky", pattern: "01011", label: "Index, ring + pinky", emoji: "🖖", description: "Three fingers up" },
  { id: "Victory", pattern: "01100", label: "Two", emoji: "✌️", description: "Peace sign" },
  { id: "Peace_Pinky", pattern: "01101", label: "Two + pinky", emoji: "🖖", description: "Index, middle and pinky up" },
  { id: "Three_Fingers", pattern: "01110", label: "Three", emoji: "3️⃣", description: "Index, middle and ring up" },
  { id: "Four_Fingers", pattern: "01111", label: "Four", emoji: "🖐️", description: "Four fingers up" },
  { id: "Thumb_Up", pattern: "10000", label: "Like", emoji: "👍", description: "Thumb up" },
  { id: "Call_Me", pattern: "10001", label: "Call me", emoji: "🤙", description: "Thumb and pinky up" },
  { id: "Thumb_Ring", pattern: "10010", label: "Thumb + ring", emoji: "👍", description: "Thumb and ring up" },
  { id: "Thumb_Ring_Pinky", pattern: "10011", label: "Thumb, ring + pinky", emoji: "🤙", description: "Thumb, ring and pinky up" },
  { id: "Thumb_Middle_Pinky", pattern: "10101", label: "Thumb, middle + pinky", emoji: "🤟", description: "Thumb, middle and pinky up" },
  { id: "Thumb_Middle_Ring_Pinky", pattern: "10111", label: "Thumb + three fingers", emoji: "🖐️", description: "Thumb, middle, ring and pinky up" },
  { id: "L_Sign", pattern: "11000", label: "L sign", emoji: "👆", description: "Thumb and index up" },
  { id: "ILoveYou", pattern: "11001", label: "Love", emoji: "🤟", description: "I love you sign" },
  { id: "Thumb_Index_Ring", pattern: "11010", label: "Thumb, index + ring", emoji: "🖖", description: "Thumb, index and ring up" },
  { id: "Thumb_Index_Ring_Pinky", pattern: "11011", label: "Thumb, index, ring + pinky", emoji: "🖐️", description: "Four fingers with middle curled" },
  { id: "Thumb_Index_Middle", pattern: "11100", label: "Thumb, index + middle", emoji: "🤟", description: "Thumb, index and middle up" },
  { id: "Thumb_Index_Middle_Pinky", pattern: "11101", label: "Thumb, index, middle + pinky", emoji: "🖐️", description: "Four fingers with ring curled" },
  { id: "Thumb_Four", pattern: "11110", label: "Thumb + four fingers", emoji: "🖐️", description: "Thumb, index, middle and ring up" },
  { id: "Open_Palm", pattern: "11111", label: "Stop", emoji: "✋", description: "Open palm" },
  { id: "Thumb_Down", label: "Nope", emoji: "👎", description: "Thumb down" },
  { id: "Pinch", label: "Pinch", emoji: "🤏", description: "Touch thumb and index" },
  { id: "OK_Sign", label: "OK", emoji: "👌", description: "Make an OK circle" },
];
const gestureDetails = Object.fromEntries(
  gestureCatalog.map(({ id, label, emoji }) => [id, { label, emoji }]),
);
const gestureByPattern = new Map(
  gestureCatalog.filter((gesture) => gesture.pattern).map((gesture) => [gesture.pattern, gesture.id]),
);
const gestureColors = ["emoji-lilac", "emoji-peach", "emoji-mint", "emoji-pink", "emoji-yellow", "emoji-blue", "emoji-violet"];

const video = document.querySelector("#camera-video");
const canvas = document.querySelector("#landmark-canvas");
const context = canvas.getContext("2d");
const placeholder = document.querySelector("#camera-placeholder");
const button = document.querySelector("#camera-button");
const buttonLabel = document.querySelector("#button-label");
const status = document.querySelector("#camera-status");
const statusText = document.querySelector("#status-text");
const cameraMessage = document.querySelector("#camera-message");
const cameraHint = document.querySelector("#camera-hint");
const detectionPill = document.querySelector("#detection-pill");
const detectedEmoji = document.querySelector("#detected-emoji");
const detectedLabel = document.querySelector("#detected-label");
const toast = document.querySelector("#toast");
const landmarkToggle = document.querySelector("#landmark-toggle");
const sessionDuration = document.querySelector("#session-duration");
const gestureCount = document.querySelector("#gesture-count");
const uniqueCount = document.querySelector("#unique-count");
const gestureHistory = document.querySelector("#gesture-history");
const exportHistoryButton = document.querySelector("#export-history");
const resetSessionButton = document.querySelector("#reset-session");
const gestureList = document.querySelector("#gesture-list");
const gestureTotal = document.querySelector("#gesture-total");

let recognizer;
let stream;
let animationFrame;
let previousVideoTime = -1;
let candidateGesture = "";
let candidateFrames = 0;
let activeGesture = "";
let toastTimer;
let sessionStartedAt;
let sessionElapsedMs = 0;
let sessionTimer;
let sessionEvents = [];
const sessionGestures = new Set();

function showError(message) {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 6500);
}

function setLoading(isLoading) {
  button.disabled = isLoading;
  buttonLabel.textContent = isLoading ? "Waking up the magic..." : "Start camera";
  cameraMessage.textContent = isLoading
    ? "Loading hand recognition. This can take a moment."
    : "Good lighting makes the magic happen.";
  statusText.textContent = isLoading ? "GETTING READY" : "CAMERA OFF";
  status.classList.remove("is-live");
}

function resetGesture() {
  candidateGesture = "";
  candidateFrames = 0;
  activeGesture = "";
  detectionPill.hidden = true;
  cameraHint.hidden = false;
}

function formatDuration(milliseconds) {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function updateSessionDuration() {
  const elapsed = sessionElapsedMs + (sessionStartedAt ? Date.now() - sessionStartedAt : 0);
  sessionDuration.textContent = formatDuration(elapsed);
}

function stopSessionClock() {
  if (sessionStartedAt) {
    sessionElapsedMs += Date.now() - sessionStartedAt;
    sessionStartedAt = undefined;
  }
  window.clearInterval(sessionTimer);
  sessionTimer = undefined;
  updateSessionDuration();
}

function beginSessionClock() {
  sessionStartedAt = Date.now();
  updateSessionDuration();
  if (!sessionTimer) {
    sessionTimer = window.setInterval(updateSessionDuration, 1000);
  }
}

function renderSessionHistory() {
  gestureCount.textContent = String(sessionEvents.length);
  uniqueCount.textContent = String(sessionGestures.size);
  exportHistoryButton.disabled = sessionEvents.length === 0;
  gestureHistory.replaceChildren();

  if (sessionEvents.length === 0) {
    const emptyMessage = document.createElement("li");
    emptyMessage.className = "history-empty";
    emptyMessage.textContent = "Detected gestures will appear here.";
    gestureHistory.append(emptyMessage);
    return;
  }

  for (const event of sessionEvents.slice(-8).reverse()) {
    const row = document.createElement("li");
    row.className = "history-item";

    const emoji = document.createElement("span");
    emoji.className = "history-emoji";
    emoji.setAttribute("aria-hidden", "true");
    emoji.textContent = event.emoji;

    const label = document.createElement("span");
    label.className = "history-label";
    label.textContent = event.label;

    const time = document.createElement("time");
    time.className = "history-time";
    time.dateTime = event.timestamp.toISOString();
    time.textContent = event.timestamp.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    row.append(emoji, label, time);
    gestureHistory.append(row);
  }
}

function renderGestureGuide() {
  gestureTotal.textContent = String(gestureCatalog.length);

  gestureCatalog.forEach((gesture, index) => {
    const row = document.createElement("div");
    row.className = "gesture-row";

    const emoji = document.createElement("span");
    emoji.className = `gesture-emoji ${gestureColors[index % gestureColors.length]}`;
    emoji.setAttribute("aria-hidden", "true");
    emoji.textContent = gesture.emoji;

    const name = document.createElement("span");
    name.className = "gesture-name";
    name.textContent = gesture.label;

    const description = document.createElement("span");
    description.className = "gesture-description";
    description.textContent = gesture.description;

    const arrow = document.createElement("span");
    arrow.className = "gesture-arrow";
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "↗";

    row.append(emoji, name, description, arrow);
    gestureList.append(row);
  });
}

function recordGesture(gestureName, gesture) {
  sessionEvents.push({
    gestureName,
    label: gesture.label,
    emoji: gesture.emoji,
    timestamp: new Date(),
  });
  sessionGestures.add(gestureName);
  renderSessionHistory();
}

function exportSessionHistory() {
  const escapeCsv = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const rows = [
    ["Time", "Gesture", "Gesture ID"],
    ...sessionEvents.map((event) => [
      event.timestamp.toISOString(),
      event.label,
      event.gestureName,
    ]),
  ];
  const csv = rows.map((row) => row.map(escapeCsv).join(",")).join("\r\n");
  const downloadUrl = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = `gesture-session-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
}

function stopCamera() {
  stopSessionClock();
  window.cancelAnimationFrame(animationFrame);
  animationFrame = undefined;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = undefined;
  }
  recognizer?.close();
  recognizer = undefined;
  video.pause();
  video.srcObject = null;
  video.style.display = "none";
  placeholder.hidden = false;
  context.clearRect(0, 0, canvas.width, canvas.height);
  canvas.width = 0;
  canvas.height = 0;
  status.classList.remove("is-live");
  statusText.textContent = "CAMERA OFF";
  button.classList.remove("is-stop");
  buttonLabel.textContent = "Start camera";
  button.querySelector(".button-icon").textContent = "◉";
  cameraMessage.textContent = "Good lighting makes the magic happen.";
  button.disabled = false;
  resetGesture();
}

function updateDetection(gestureName) {
  const recognizedGesture = gestureDetails[gestureName];
  if (!recognizedGesture) {
    candidateGesture = "";
    candidateFrames = 0;
    activeGesture = "";
    detectionPill.hidden = true;
    cameraHint.hidden = false;
    return;
  }

  if (candidateGesture === gestureName) {
    candidateFrames += 1;
  } else {
    candidateGesture = gestureName;
    candidateFrames = 1;
  }

  if (candidateFrames < STABLE_FRAMES) return;

  if (activeGesture !== gestureName) {
    recordGesture(gestureName, recognizedGesture);
  }
  activeGesture = gestureName;
  detectedEmoji.textContent = recognizedGesture.emoji;
  detectedLabel.textContent = recognizedGesture.label;
  detectionPill.hidden = false;
  cameraHint.hidden = true;
}

function distanceBetween(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function detectCustomGesture(hand) {
  const handSize = distanceBetween(hand[0], hand[9]);

  if (handSize === 0) return "";

  const thumbIndexDistance = distanceBetween(hand[4], hand[8]) / handSize;
  const fingerIsExtended = (tipIndex, jointIndex) =>
    distanceBetween(hand[tipIndex], hand[0]) >
    distanceBetween(hand[jointIndex], hand[0]) + handSize * 0.06;
  const indexExtended = fingerIsExtended(8, 6);
  const middleExtended = fingerIsExtended(12, 10);
  const ringExtended = fingerIsExtended(16, 14);
  const pinkyExtended = fingerIsExtended(20, 18);
  const thumbExtended = distanceBetween(hand[4], hand[5]) > handSize * 0.42;
  const otherFingersExtended = [indexExtended, middleExtended, ringExtended, pinkyExtended];

  if (thumbIndexDistance < 0.24) {
    if (otherFingersExtended.every(Boolean)) return "OK_Sign";
    return "Pinch";
  }

  if (
    thumbExtended &&
    hand[4].y > hand[3].y + handSize * 0.12 &&
    otherFingersExtended.every((extended) => !extended)
  ) {
    return "Thumb_Down";
  }

  const pattern = [
    thumbExtended,
    indexExtended,
    middleExtended,
    ringExtended,
    pinkyExtended,
  ]
    .map(Number)
    .join("");
  return gestureByPattern.get(pattern) ?? "";
}

function drawLandmarks(landmarks) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarkToggle.checked) return;

  for (const hand of landmarks) {
    for (const point of hand) {
      context.beginPath();
      context.arc(
        point.x * canvas.width,
        point.y * canvas.height,
        Math.max(3, canvas.width * 0.006),
        0,
        Math.PI * 2,
      );
      context.fillStyle = "#d9ff67";
      context.shadowColor = "#d9ff67";
      context.shadowBlur = 10;
      context.fill();
    }
  }
  context.shadowBlur = 0;
}

function detectFrame() {
  if (!stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animationFrame = window.requestAnimationFrame(detectFrame);
    return;
  }

  if (video.currentTime !== previousVideoTime) {
    previousVideoTime = video.currentTime;
    const result = recognizer.recognizeForVideo(video, performance.now());
    drawLandmarks(result.landmarks ?? []);

    const topGesture = result.gestures?.[0]?.[0];
    const customGesture = result.landmarks?.[0]
      ? detectCustomGesture(result.landmarks[0])
      : "";
    const confidentModelGesture =
      topGesture && topGesture.score >= MIN_CONFIDENCE
        ? topGesture.categoryName
        : "";
    updateDetection(
      confidentModelGesture === "Thumb_Down"
        ? confidentModelGesture
        : customGesture || confidentModelGesture,
    );
  }

  animationFrame = window.requestAnimationFrame(detectFrame);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Camera access needs a supported browser and a secure connection (HTTPS or localhost).");
    return;
  }

  setLoading(true);
  try {
    const cameraPromise = navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 960 },
          height: { ideal: 540 },
        },
      })
      .then((cameraStream) => {
        stream = cameraStream;
        return cameraStream;
      });
    const [fileset] = await Promise.all([
      FilesetResolver.forVisionTasks(WASM_URL),
      cameraPromise,
    ]);
    recognizer = await GestureRecognizer.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: MIN_CONFIDENCE,
      minHandPresenceConfidence: MIN_CONFIDENCE,
      minTrackingConfidence: MIN_CONFIDENCE,
    });

    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    placeholder.hidden = true;
    video.style.display = "block";
    status.classList.add("is-live");
    statusText.textContent = "LIVE · ON DEVICE";
    button.classList.add("is-stop");
    buttonLabel.textContent = "Stop camera";
    button.querySelector(".button-icon").textContent = "■";
    cameraMessage.textContent = "Nice! Show your hand to the camera.";
    button.disabled = false;
    previousVideoTime = -1;
    beginSessionClock();
    animationFrame = window.requestAnimationFrame(detectFrame);
  } catch (error) {
    stopCamera();
    if (error instanceof DOMException && error.name === "NotAllowedError") {
      showError("Camera permission was blocked. Allow camera access in your browser settings, then try again.");
    } else if (error instanceof DOMException && error.name === "NotFoundError") {
      showError("No camera was found. Connect a camera and try again.");
    } else {
      console.error("Unable to start hand gesture recognition:", error);
      showError("Could not start the camera or load gesture recognition. Check your connection and try again.");
    }
  }
}

button.addEventListener("click", () => {
  if (stream) {
    stopCamera();
    return;
  }
  startCamera();
});

renderGestureGuide();
renderSessionHistory();

landmarkToggle.addEventListener("change", () => {
  if (!landmarkToggle.checked) {
    context.clearRect(0, 0, canvas.width, canvas.height);
  }
});

exportHistoryButton.addEventListener("click", exportSessionHistory);

resetSessionButton.addEventListener("click", () => {
  sessionEvents = [];
  sessionGestures.clear();
  sessionElapsedMs = 0;
  if (sessionStartedAt) {
    sessionStartedAt = Date.now();
  }
  updateSessionDuration();
  renderSessionHistory();
});

window.addEventListener("pagehide", stopCamera);
