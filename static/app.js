let currentSession = null;
let videoStream = null;
let capturedFrames = [];
let capturedIdFile = null;

const API_KEY = "trustid_live_demo_key_2026";

const videoElement = document.getElementById("webcam-video");
const canvasElement = document.getElementById("canvas-overlay");
const canvasCtx = canvasElement.getContext("2d");

const btnStartCam = document.getElementById("btn-start-camera");
const btnInitSession = document.getElementById("btn-init-session");
const btnCapture = document.getElementById("btn-capture-frames");
const btnSubmit = document.getElementById("btn-submit-verify");
const challengeBanner = document.getElementById("challenge-banner");
const challengeText = document.getElementById("challenge-text");

// Initialize Session
btnInitSession.addEventListener("click", async () => {
  btnInitSession.disabled = true;
  btnInitSession.innerText = "Initializing...";

  try {
    const res = await fetch("/api/v1/sessions/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-KEY": API_KEY,
      },
      body: JSON.stringify({
        user_id: "demo_applicant_" + Math.floor(Math.random() * 90000 + 10000),
        document_type: "AADHAAR",
      }),
    });

    if (!res.ok) throw new Error("Failed to create session");
    currentSession = await res.json();

    document.getElementById("session-id-display").innerText = currentSession.session_id;
    document.getElementById("user-id-display").innerText = currentSession.user_id;

    const instructions = currentSession.challenges.map(c => c.instruction).join(" ➔ ");
    challengeText.innerText = `Challenges: ${instructions} | Voice Code: ${currentSession.verification_code}`;
    challengeBanner.style.display = "flex";

    btnInitSession.innerText = "Session Active";
    btnCapture.disabled = false;
  } catch (err) {
    alert("Error initializing session: " + err.message);
    btnInitSession.disabled = false;
    btnInitSession.innerText = "Initialize Session";
  }
});

// Start Camera
btnStartCam.addEventListener("click", async () => {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    videoElement.srcObject = videoStream;
    btnStartCam.innerText = "Camera Connected";
    btnStartCam.disabled = true;
  } catch (err) {
    console.warn("Webcam access error or denied, using simulated generator:", err);
    alert("Webcam permission denied or unavailable. You can upload an image or test with demo frames.");
  }
});

// Capture Frames
btnCapture.addEventListener("click", async () => {
  if (!currentSession) {
    alert("Please initialize a session first.");
    return;
  }

  btnCapture.disabled = true;
  btnCapture.innerText = "Capturing 3 Frames...";
  capturedFrames = [];

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 640;
  tempCanvas.height = 480;
  const ctx = tempCanvas.getContext("2d");

  // Capture 3 frames with 600ms gap
  for (let i = 0; i < 3; i++) {
    if (videoElement.srcObject) {
      ctx.drawImage(videoElement, 0, 0, 640, 480);
    } else {
      // Synthetic canvas generator for testing if camera absent
      ctx.fillStyle = "#1e293b";
      ctx.fillRect(0, 0, 640, 480);
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(320, 240, 100, 0, Math.PI * 2);
      ctx.fill();
    }

    const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, "image/jpeg", 0.9));
    capturedFrames.push(blob);
    await new Promise(r => setTimeout(r, 600));
  }

  btnCapture.innerText = `Captured (${capturedFrames.length} Frames)`;
  btnSubmit.disabled = false;
});

// ID Card File Input
const idInput = document.getElementById("id-card-input");
idInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    capturedIdFile = e.target.files[0];
    document.getElementById("id-file-name").innerText = capturedIdFile.name;
  }
});

// Submit Full Verification
btnSubmit.addEventListener("click", async () => {
  if (!currentSession) {
    alert("Please create a session.");
    return;
  }

  btnSubmit.disabled = true;
  btnSubmit.innerText = "Running AI Models...";

  const formData = new FormData();
  formData.append("session_id", currentSession.session_id);

  // ID Document
  if (capturedIdFile) {
    formData.append("id_document", capturedIdFile);
  } else {
    // Generate synthetic mock ID document image
    const idCanvas = document.createElement("canvas");
    idCanvas.width = 600;
    idCanvas.height = 380;
    const ictx = idCanvas.getContext("2d");
    ictx.fillStyle = "#ffffff";
    ictx.fillRect(0, 0, 600, 380);
    ictx.fillStyle = "#0284c7";
    ictx.fillRect(0, 0, 600, 40);
    ictx.fillStyle = "#1e293b";
    ictx.font = "20px Arial";
    ictx.fillText("GOVERNMENT OF INDIA - AADHAAR", 120, 28);
    ictx.fillText("Name: Rahul Sharma", 220, 120);
    ictx.fillText("DOB: 15/08/1994", 220, 160);
    ictx.fillText("Gender: MALE", 220, 200);
    ictx.font = "bold 22px Arial";
    ictx.fillText("9841 2341 5678", 220, 270);
    // ID Photo box
    ictx.fillStyle = "#94a3b8";
    ictx.fillRect(30, 80, 150, 180);

    const idBlob = await new Promise(resolve => idCanvas.toBlob(resolve, "image/jpeg", 0.95));
    formData.append("id_document", idBlob, "mock_aadhaar.jpg");
  }

  // Live Frames
  if (capturedFrames.length === 0) {
    // Single fallback frame
    const fCanvas = document.createElement("canvas");
    fCanvas.width = 640;
    fCanvas.height = 480;
    const fctx = fCanvas.getContext("2d");
    fctx.fillStyle = "#334155";
    fctx.fillRect(0, 0, 640, 480);
    const fb = await new Promise(resolve => fCanvas.toBlob(resolve, "image/jpeg", 0.9));
    formData.append("live_frames", fb, "live_frame_0.jpg");
  } else {
    capturedFrames.forEach((blob, idx) => {
      formData.append("live_frames", blob, `live_frame_${idx}.jpg`);
    });
  }

  try {
    const res = await fetch("/api/v1/verify/full", {
      method: "POST",
      headers: {
        "X-API-KEY": API_KEY,
      },
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Verification failed");
    }

    const data = await res.json();
    renderResults(data);
    btnSubmit.innerText = "Completed";
  } catch (err) {
    alert("Verification error: " + err.message);
    btnSubmit.disabled = false;
    btnSubmit.innerText = "Run Full VKYC Verification";
  }
});

function renderResults(data) {
  const dossier = data.dossier;

  // Verdict Banner
  const verdictBanner = document.getElementById("verdict-banner");
  verdictBanner.innerText = `VERDICT: ${dossier.verdict}`;
  verdictBanner.className = "verdict-banner";
  if (dossier.verdict === "APPROVED") {
    verdictBanner.classList.add("verdict-approved");
  } else if (dossier.verdict === "REVIEW_REQUIRED") {
    verdictBanner.classList.add("verdict-review");
  } else {
    verdictBanner.classList.add("verdict-rejected");
  }

  // Face Match Score
  const faceScore = Math.round(dossier.face_match.match_score * 100);
  document.getElementById("score-face").innerText = `${faceScore}%`;
  document.getElementById("bar-face").style.width = `${faceScore}%`;

  // Liveness Score
  const livenessScore = Math.round(dossier.liveness.passive_score * 100);
  document.getElementById("score-liveness").innerText = `${livenessScore}%`;
  document.getElementById("bar-liveness").style.width = `${livenessScore}%`;

  // Deepfake Risk Score (Lower is safer)
  const dfRisk = Math.round(dossier.deepfake.synthetic_risk_score * 100);
  document.getElementById("score-deepfake").innerText = `${dfRisk}% (Risk)`;
  document.getElementById("bar-deepfake").style.width = `${dfRisk}%`;

  // Document OCR Summary
  const doc = dossier.document;
  document.getElementById("doc-number-val").innerText = doc.extracted_data.document_number || "Verified";
  document.getElementById("doc-name-val").innerText = doc.extracted_data.full_name || "Rahul Sharma";
  document.getElementById("doc-type-val").innerText = doc.document_type;

  // Cryptographic Audit Hash
  document.getElementById("audit-hash-display").innerText = dossier.tamper_proof_audit_hash;
  document.getElementById("audit-sig-display").innerText = dossier.signature;

  // Results Section Reveal
  document.getElementById("results-panel").style.display = "block";
}
