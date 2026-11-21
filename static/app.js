// TrustID developer demo portal.
// The API key is typed in by the developer and kept in sessionStorage for this
// tab only. Real integrations create sessions from a backend.

let currentSession = null;
let videoStream = null;
let capturedFrames = [];
let capturedIdFile = null;

const FRAME_W = 640;
const FRAME_H = 480;
const NEUTRAL_FRAMES = 6;
const FRAMES_PER_CHALLENGE = 8;
const FRAME_INTERVAL_MS = 180;
const PREP_DELAY_MS = 1200;

const $ = (id) => document.getElementById(id);
const videoElement = $("webcam-video");
const apiKeyInput = $("api-key-input");
const btnStartCam = $("btn-start-camera");
const btnInitSession = $("btn-init-session");
const btnCapture = $("btn-capture-frames");
const btnSubmit = $("btn-submit-verify");
const challengeBanner = $("challenge-banner");
const challengeText = $("challenge-text");

try {
  apiKeyInput.value = sessionStorage.getItem("trustid_api_key") || "";
} catch (_) { /* storage unavailable */ }
apiKeyInput.addEventListener("change", () => {
  try { sessionStorage.setItem("trustid_api_key", apiKeyInput.value.trim()); } catch (_) {}
});

function apiKey() {
  const key = apiKeyInput.value.trim();
  if (!key) throw new Error("Enter an API key first.");
  return key;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function showBanner(text) {
  challengeText.textContent = text;
  challengeBanner.style.display = "flex";
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail || body);
  } catch (_) {
    return `HTTP ${res.status}`;
  }
}

// 1. Session
btnInitSession.addEventListener("click", async () => {
  btnInitSession.disabled = true;
  btnInitSession.textContent = "Initializing...";
  try {
    const res = await fetch("/api/v1/sessions/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-KEY": apiKey() },
      body: JSON.stringify({
        user_id: "demo_applicant_" + Math.floor(Math.random() * 90000 + 10000),
        document_type: "AADHAAR",
      }),
    });
    if (!res.ok) throw new Error(await errorMessage(res));
    currentSession = await res.json();

    $("session-id-display").textContent = currentSession.session_id;
    $("user-id-display").textContent = currentSession.user_id;
    const steps = currentSession.challenges.map((c) => c.instruction).join("  ➔  ");
    showBanner(`Challenges: ${steps}`);

    btnInitSession.textContent = "Session Active";
    btnCapture.disabled = false;
  } catch (err) {
    alert("Could not create session: " + err.message);
    btnInitSession.disabled = false;
    btnInitSession.textContent = "1. Initialize Session";
  }
});

// Camera
btnStartCam.addEventListener("click", async () => {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: FRAME_W, height: FRAME_H, facingMode: "user" },
      audio: false,
    });
    videoElement.srcObject = videoStream;
    btnStartCam.textContent = "Camera Connected";
    btnStartCam.disabled = true;
  } catch (err) {
    alert("Camera unavailable or permission denied. A live camera is required for liveness checks.");
  }
});

// 2. Guided capture: neutral frames first, then a burst per challenge.
async function captureBurst(ctx, canvas, count) {
  for (let i = 0; i < count; i++) {
    ctx.drawImage(videoElement, 0, 0, FRAME_W, FRAME_H); // unmirrored, as the API expects
    capturedFrames.push(await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.85)));
    await sleep(FRAME_INTERVAL_MS);
  }
}

btnCapture.addEventListener("click", async () => {
  if (!currentSession) return alert("Initialize a session first.");
  if (!videoElement.srcObject) return alert("Connect the camera first.");

  btnCapture.disabled = true;
  capturedFrames = [];
  const canvas = document.createElement("canvas");
  canvas.width = FRAME_W;
  canvas.height = FRAME_H;
  const ctx = canvas.getContext("2d");

  showBanner("Look straight at the camera with a neutral face");
  await sleep(PREP_DELAY_MS);
  await captureBurst(ctx, canvas, NEUTRAL_FRAMES);

  for (const step of currentSession.challenges) {
    showBanner(`Now: ${step.instruction}`);
    await sleep(PREP_DELAY_MS);
    await captureBurst(ctx, canvas, FRAMES_PER_CHALLENGE);
  }

  showBanner(`Captured ${capturedFrames.length} frames. Upload the ID image, then run verification.`);
  btnCapture.textContent = `Captured (${capturedFrames.length} frames)`;
  btnSubmit.disabled = false;
});

$("id-card-input").addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    capturedIdFile = e.target.files[0];
    $("id-file-name").textContent = capturedIdFile.name;
  }
});

// 3. Submit
btnSubmit.addEventListener("click", async () => {
  if (!currentSession) return alert("Create a session first.");
  if (!capturedIdFile) return alert("Upload an ID document image first.");
  if (capturedFrames.length === 0) return alert("Capture live frames first.");

  btnSubmit.disabled = true;
  btnSubmit.textContent = "Running checks...";

  const formData = new FormData();
  formData.append("session_id", currentSession.session_id);
  formData.append("id_document", capturedIdFile);
  capturedFrames.forEach((blob, i) => formData.append("live_frames", blob, `live_frame_${i}.jpg`));
  const ocrText = $("ocr-text-input").value.trim();
  if (ocrText) formData.append("extracted_ocr_text", ocrText);

  try {
    const res = await fetch("/api/v1/verify/full", {
      method: "POST",
      headers: { "X-API-KEY": apiKey() },
      body: formData,
    });
    if (!res.ok) throw new Error(await errorMessage(res));
    renderResults(await res.json());
    btnSubmit.textContent = "Completed (create a new session to retry)";
  } catch (err) {
    alert("Verification error: " + err.message);
    btnSubmit.disabled = false;
    btnSubmit.textContent = "🚀 3. Run Full VKYC Verification";
  }
});

function setBar(scoreId, barId, value, suffix = "") {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  $(scoreId).textContent = `${pct}%${suffix}`;
  $(barId).style.width = `${pct}%`;
}

function renderResults(data) {
  const d = data.dossier;

  const banner = $("verdict-banner");
  banner.textContent = `VERDICT: ${d.verdict}`;
  banner.className = "verdict-banner " + (
    d.verdict === "APPROVED" ? "verdict-approved" : d.verdict === "REVIEW_REQUIRED" ? "verdict-review" : "verdict-rejected"
  );

  setBar("score-face", "bar-face", d.face_match.match_score, ` (threshold ${Math.round(d.face_match.threshold * 100)}%)`);
  setBar("score-liveness", "bar-liveness", d.liveness.passive_score);
  setBar("score-deepfake", "bar-deepfake", d.deepfake.synthetic_risk_score, " (risk)");

  const doc = d.document;
  $("doc-type-val").textContent = doc.document_type;
  $("doc-number-val").textContent = doc.extracted_data.document_number || "not detected";
  $("doc-name-val").textContent = doc.extracted_data.full_name || "not detected";
  $("doc-source-val").textContent = doc.text_source;

  const list = $("reasons-list");
  list.replaceChildren();
  const reasons = [
    ...d.rejection_reasons.map((r) => `Rejected: ${r}`),
    ...d.review_reasons.map((r) => `Review: ${r}`),
  ];
  if (reasons.length === 0) reasons.push("All checks passed.");
  for (const text of reasons) {
    const li = document.createElement("li");
    li.textContent = text;
    list.appendChild(li);
  }

  $("audit-hash-display").textContent = d.tamper_proof_audit_hash;
  $("audit-sig-display").textContent = d.signature;
  $("results-panel").style.display = "block";
  const placeholder = $("results-placeholder");
  if (placeholder) placeholder.style.display = "none";
}
