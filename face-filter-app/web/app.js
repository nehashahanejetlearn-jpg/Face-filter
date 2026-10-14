/**
 * VisionAR - Real-time 468-Point Neural Face Filter Studio
 * 11 Features: Sunglasses, Mustache, Crown, Puppy, Clown, Laser Eyes,
 * Cyber Mesh, Cartoon, VHS Glitch, Alien Big Eyes, and HD Snapshot.
 */

// DOM Elements
const videoElement = document.getElementById('inputVideo');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');
const trackingStatus = document.getElementById('trackingStatus');
const statusText = trackingStatus.querySelector('.status-text');
const fpsDisplay = document.getElementById('fpsDisplay');
const activeFilterTag = document.getElementById('activeFilterTag');
const btnFlipCamera = document.getElementById('btnFlipCamera');
const btnResetFilters = document.getElementById('btnResetFilters');
const btnSnapshot = document.getElementById('btnSnapshot');
const flashOverlay = document.getElementById('flashOverlay');
const countdownOverlay = document.getElementById('countdownOverlay');
const snapshotGallery = document.getElementById('snapshotGallery');
const multiFilterCheck = document.getElementById('multiFilterCheck');
const filterCards = document.querySelectorAll('.filter-card');

// State
let activeFilters = new Set(['sunglasses']);
let isMirrored = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;
let isSnapping = false;

// MediaPipe Landmark Index Shortcuts
// Left eye: 33 (outer), 133 (inner), 159 (top), 145 (bottom), 468 (iris center)
// Right eye: 362 (inner), 263 (outer), 386 (top), 374 (bottom), 473 (iris center)
// Nose: 4 (tip), 2 (bottom), 168 (bridge)
// Mouth: 0 (upper lip top), 17 (lower lip bottom), 61 (left corner), 291 (right corner)
// Head: 10 (top forehead), 152 (chin), 234 (left cheek), 454 (right cheek)
const LM = {
  noseTip: 4,
  noseBottom: 2,
  noseBridge: 168,
  leftEyeCenter: 468,
  rightEyeCenter: 473,
  leftEyeOuter: 33,
  rightEyeOuter: 263,
  foreheadTop: 10,
  chin: 152,
  leftEar: 234,
  rightEar: 454,
  upperLip: 0
};

// -------------------------------------------------------------
// Initialize MediaPipe FaceMesh
// -------------------------------------------------------------
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
  maxNumFaces: 2,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

faceMesh.onResults(onResults);

// Setup Webcam
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceMesh.send({ image: videoElement });
  },
  width: 1280,
  height: 720
});

camera.start()
  .then(() => {
    trackingStatus.classList.add('ready');
    statusText.textContent = 'AI Face Tracking Active';
  })
  .catch((err) => {
    console.error('Camera initialization failed:', err);
    statusText.textContent = 'Camera Access Blocked';
  });

// -------------------------------------------------------------
// Face Mesh Results & Rendering Loop
// -------------------------------------------------------------
function onResults(results) {
  // Update FPS
  const now = performance.now();
  frameCount++;
  if (now - lastFrameTime >= 1000) {
    currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime));
    fpsDisplay.textContent = `${currentFps} FPS`;
    frameCount = 0;
    lastFrameTime = now;
  }

  // Ensure canvas dimensions match video
  if (canvasElement.width !== results.image.width || canvasElement.height !== results.image.height) {
    canvasElement.width = results.image.width;
    canvasElement.height = results.image.height;
  }

  const w = canvasElement.width;
  const h = canvasElement.height;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, w, h);

  // 1. Draw base video feed (with optional full-frame filters like Cartoon or VHS)
  if (activeFilters.has('cartoon')) {
    drawCartoonVideo(results.image, w, h);
  } else {
    canvasCtx.drawImage(results.image, 0, 0, w, h);
  }

  // 2. Render face-anchored filters if faces detected
  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    statusText.textContent = `Tracking (${results.multiFaceLandmarks.length} Face${results.multiFaceLandmarks.length > 1 ? 's' : ''})`;
    
    for (const rawLandmarks of results.multiFaceLandmarks) {
      // Convert normalized landmarks (0-1) to pixel coords
      const lm = rawLandmarks.map(p => ({
        x: p.x * w,
        y: p.y * h,
        z: p.z * w
      }));

      // Render Active Filters in Order
      if (activeFilters.has('alien')) drawAlienMorph(results.image, lm, w, h);
      if (activeFilters.has('facemesh')) draw3DFaceMesh(rawLandmarks, w, h);
      if (activeFilters.has('sunglasses')) drawSunglasses(lm);
      if (activeFilters.has('mustache')) drawMustache(lm);
      if (activeFilters.has('crown')) drawCrown(lm);
      if (activeFilters.has('puppy')) drawPuppy(lm);
      if (activeFilters.has('clown')) drawClownNose(lm);
      if (activeFilters.has('lasereyes')) drawLaserEyes(lm);
    }
  } else {
    statusText.textContent = 'Searching for Face...';
  }

  // 3. Post-process Screen Filters (VHS Glitch overlay)
  if (activeFilters.has('vhs')) {
    drawVhsGlitch(w, h);
  }

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 1: Cool Sunglasses / Thug Life Shades
// -------------------------------------------------------------
function drawSunglasses(lm) {
  const leftEye = lm[LM.leftEyeCenter] || lm[33];
  const rightEye = lm[LM.rightEyeCenter] || lm[263];

  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const eyeDistance = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);
  const centerX = (leftEye.x + rightEye.x) / 2;
  const centerY = (leftEye.y + rightEye.y) / 2;

  canvasCtx.save();
  canvasCtx.translate(centerX, centerY);
  canvasCtx.rotate(angle);

  const glassWidth = eyeDistance * 2.3;
  const glassHeight = eyeDistance * 0.9;
  const lensWidth = glassWidth * 0.44;
  const bridgeGap = glassWidth * 0.12;

  // Frame Bridge
  canvasCtx.fillStyle = '#0f172a';
  canvasCtx.fillRect(-bridgeGap / 2, -glassHeight * 0.25, bridgeGap, glassHeight * 0.15);

  // Left & Right Lenses (Polarized Dark)
  [-1, 1].forEach(side => {
    const lensX = side === -1 ? -bridgeGap / 2 - lensWidth : bridgeGap / 2;
    const lensY = -glassHeight * 0.4;

    // Lens Frame (Outer Shadow & Border)
    canvasCtx.save();
    canvasCtx.shadowColor = 'rgba(0,0,0,0.8)';
    canvasCtx.shadowBlur = 10;
    canvasCtx.fillStyle = '#111827';
    roundRect(canvasCtx, lensX, lensY, lensWidth, glassHeight, 8);
    canvasCtx.fill();
    canvasCtx.lineWidth = 4;
    canvasCtx.strokeStyle = '#06b6d4';
    canvasCtx.stroke();
    canvasCtx.restore();

    // Dark Lens Glass with Glossy Gradient
    const grad = canvasCtx.createLinearGradient(lensX, lensY, lensX + lensWidth, lensY + glassHeight);
    grad.addColorStop(0, 'rgba(30, 41, 59, 0.95)');
    grad.addColorStop(0.5, 'rgba(15, 23, 42, 0.98)');
    grad.addColorStop(1, 'rgba(2, 6, 23, 1)');
    canvasCtx.fillStyle = grad;
    roundRect(canvasCtx, lensX + 3, lensY + 3, lensWidth - 6, glassHeight - 6, 6);
    canvasCtx.fill();

    // Specular Reflection Streak
    canvasCtx.save();
    canvasCtx.beginPath();
    canvasCtx.moveTo(lensX + lensWidth * 0.15, lensY + glassHeight * 0.15);
    canvasCtx.lineTo(lensX + lensWidth * 0.45, lensY + glassHeight * 0.85);
    canvasCtx.lineWidth = 4;
    canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    canvasCtx.stroke();

    canvasCtx.beginPath();
    canvasCtx.moveTo(lensX + lensWidth * 0.3, lensY + glassHeight * 0.15);
    canvasCtx.lineTo(lensX + lensWidth * 0.55, lensY + glassHeight * 0.85);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    canvasCtx.stroke();
    canvasCtx.restore();
  });

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 2: Vintage Gentleman Mustache
// -------------------------------------------------------------
function drawMustache(lm) {
  const noseBottom = lm[LM.noseBottom];
  const upperLip = lm[LM.upperLip];
  const leftCheek = lm[LM.leftEar];
  const rightCheek = lm[LM.rightEar];

  const dx = rightCheek.x - leftCheek.x;
  const dy = rightCheek.y - leftCheek.y;
  const angle = Math.atan2(dy, dx);
  const faceWidth = Math.hypot(dx, dy);

  const centerX = (noseBottom.x + upperLip.x) / 2;
  const centerY = (noseBottom.y + upperLip.y) / 2;

  canvasCtx.save();
  canvasCtx.translate(centerX, centerY);
  canvasCtx.rotate(angle);

  const mWidth = faceWidth * 0.42;
  const mHeight = mWidth * 0.45;

  // Handlebar Mustache Path
  canvasCtx.fillStyle = '#1c1917';
  canvasCtx.beginPath();
  // Center anchor
  canvasCtx.moveTo(0, 0);
  // Left curl
  canvasCtx.bezierCurveTo(-mWidth * 0.2, -mHeight * 0.4, -mWidth * 0.45, -mHeight * 0.1, -mWidth * 0.5, -mHeight * 0.5);
  canvasCtx.bezierCurveTo(-mWidth * 0.48, mHeight * 0.6, -mWidth * 0.18, mHeight * 0.5, 0, mHeight * 0.15);
  // Right curl
  canvasCtx.bezierCurveTo(mWidth * 0.18, mHeight * 0.5, mWidth * 0.48, mHeight * 0.6, mWidth * 0.5, -mHeight * 0.5);
  canvasCtx.bezierCurveTo(mWidth * 0.45, -mHeight * 0.1, mWidth * 0.2, -mHeight * 0.4, 0, 0);
  canvasCtx.closePath();

  canvasCtx.shadowColor = 'rgba(0,0,0,0.6)';
  canvasCtx.shadowBlur = 6;
  canvasCtx.fill();

  // Subtle Highlights on curls
  canvasCtx.strokeStyle = '#44403c';
  canvasCtx.lineWidth = 2;
  canvasCtx.stroke();

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 3: Royal Golden Crown
// -------------------------------------------------------------
function drawCrown(lm) {
  const forehead = lm[LM.foreheadTop];
  const chin = lm[LM.chin];
  const leftCheek = lm[LM.leftEar];
  const rightCheek = lm[LM.rightEar];

  const dx = rightCheek.x - leftCheek.x;
  const dy = rightCheek.y - leftCheek.y;
  const angle = Math.atan2(dy, dx);
  const faceHeight = Math.hypot(chin.x - forehead.x, chin.y - forehead.y);
  const crownWidth = faceHeight * 0.95;
  const crownHeight = crownWidth * 0.65;

  canvasCtx.save();
  canvasCtx.translate(forehead.x, forehead.y - crownHeight * 0.35);
  canvasCtx.rotate(angle);

  // Crown Gradient
  const grad = canvasCtx.createLinearGradient(-crownWidth / 2, -crownHeight, crownWidth / 2, 0);
  grad.addColorStop(0, '#f59e0b');
  grad.addColorStop(0.3, '#fef08a');
  grad.addColorStop(0.7, '#fbbf24');
  grad.addColorStop(1, '#d97706');

  canvasCtx.fillStyle = grad;
  canvasCtx.strokeStyle = '#78350f';
  canvasCtx.lineWidth = 3;

  canvasCtx.beginPath();
  const half = crownWidth / 2;
  canvasCtx.moveTo(-half, 0);
  // Left spike
  canvasCtx.lineTo(-half * 0.9, -crownHeight * 0.7);
  canvasCtx.lineTo(-half * 0.45, -crownHeight * 0.4);
  // Center Grand Spike
  canvasCtx.lineTo(0, -crownHeight);
  // Right spike
  canvasCtx.lineTo(half * 0.45, -crownHeight * 0.4);
  canvasCtx.lineTo(half * 0.9, -crownHeight * 0.7);
  canvasCtx.lineTo(half, 0);
  // Base band
  canvasCtx.quadraticCurveTo(0, crownHeight * 0.15, -half, 0);
  canvasCtx.closePath();

  canvasCtx.shadowColor = 'rgba(245, 158, 11, 0.6)';
  canvasCtx.shadowBlur = 15;
  canvasCtx.fill();
  canvasCtx.stroke();

  // Jewels / Rubies on spikes
  const gems = [
    { x: -half * 0.9, y: -crownHeight * 0.7, color: '#ec4899' },
    { x: 0, y: -crownHeight, color: '#ef4444' },
    { x: half * 0.9, y: -crownHeight * 0.7, color: '#ec4899' },
    { x: 0, y: -crownHeight * 0.3, color: '#06b6d4' }
  ];

  gems.forEach(g => {
    canvasCtx.beginPath();
    canvasCtx.arc(g.x, g.y, crownWidth * 0.045, 0, Math.PI * 2);
    canvasCtx.fillStyle = g.color;
    canvasCtx.shadowColor = g.color;
    canvasCtx.shadowBlur = 10;
    canvasCtx.fill();
    canvasCtx.strokeStyle = '#fff';
    canvasCtx.lineWidth = 1.5;
    canvasCtx.stroke();
  });

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 4: Puppy Dog Ears & Nose
// -------------------------------------------------------------
function drawPuppy(lm) {
  const leftEye = lm[LM.leftEyeOuter];
  const rightEye = lm[LM.rightEyeOuter];
  const nose = lm[LM.noseTip];
  const forehead = lm[LM.foreheadTop];

  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const angle = Math.atan2(dy, dx);
  const headWidth = Math.hypot(dx, dy) * 1.8;

  canvasCtx.save();

  // Puppy Ears (Left & Right)
  [-1, 1].forEach(side => {
    canvasCtx.save();
    const earAnchorX = forehead.x + (side * headWidth * 0.45);
    const earAnchorY = forehead.y - headWidth * 0.25;

    canvasCtx.translate(earAnchorX, earAnchorY);
    canvasCtx.rotate(angle + (side * 0.25));

    const earW = headWidth * 0.35;
    const earH = headWidth * 0.65;

    // Outer brown ear
    canvasCtx.fillStyle = '#78350f';
    canvasCtx.beginPath();
    canvasCtx.ellipse(0, earH / 2, earW / 2, earH / 2, 0, 0, Math.PI * 2);
    canvasCtx.fill();

    // Inner pink ear
    canvasCtx.fillStyle = '#f472b6';
    canvasCtx.beginPath();
    canvasCtx.ellipse(0, earH / 2, earW * 0.3, earH * 0.38, 0, 0, Math.PI * 2);
    canvasCtx.fill();

    canvasCtx.restore();
  });

  // Cute Dog Nose Pad
  canvasCtx.save();
  canvasCtx.translate(nose.x, nose.y);
  canvasCtx.rotate(angle);

  const noseRadius = headWidth * 0.12;
  canvasCtx.fillStyle = '#1e1b4b';
  canvasCtx.beginPath();
  canvasCtx.ellipse(0, 0, noseRadius * 1.2, noseRadius * 0.85, 0, 0, Math.PI * 2);
  canvasCtx.fill();

  // Highlight spot
  canvasCtx.fillStyle = 'rgba(255, 255, 255, 0.7)';
  canvasCtx.beginPath();
  canvasCtx.arc(-noseRadius * 0.4, -noseRadius * 0.3, noseRadius * 0.25, 0, Math.PI * 2);
  canvasCtx.fill();
  canvasCtx.restore();

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 5: Glowing Clown Nose
// -------------------------------------------------------------
function drawClownNose(lm) {
  const nose = lm[LM.noseTip];
  const radius = Math.hypot(lm[LM.noseBottom].x - lm[LM.noseBridge].x, lm[LM.noseBottom].y - lm[LM.noseBridge].y) * 0.45;

  canvasCtx.save();
  canvasCtx.translate(nose.x, nose.y);

  // Glow halo
  canvasCtx.shadowColor = '#ef4444';
  canvasCtx.shadowBlur = 25;

  // 3D Red Sphere Gradient
  const grad = canvasCtx.createRadialGradient(
    -radius * 0.3, -radius * 0.3, radius * 0.1,
    0, 0, radius
  );
  grad.addColorStop(0, '#fca5a5');
  grad.addColorStop(0.3, '#ef4444');
  grad.addColorStop(0.85, '#b91c1c');
  grad.addColorStop(1, '#7f1d1d');

  canvasCtx.fillStyle = grad;
  canvasCtx.beginPath();
  canvasCtx.arc(0, 0, radius, 0, Math.PI * 2);
  canvasCtx.fill();

  // White Specular shine
  canvasCtx.fillStyle = 'rgba(255, 255, 255, 0.85)';
  canvasCtx.beginPath();
  canvasCtx.ellipse(-radius * 0.35, -radius * 0.35, radius * 0.28, radius * 0.18, -Math.PI / 4, 0, Math.PI * 2);
  canvasCtx.fill();

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 6: Cyberpunk Glowing Laser Eyes
// -------------------------------------------------------------
function drawLaserEyes(lm) {
  const leftEye = lm[LM.leftEyeCenter] || lm[159];
  const rightEye = lm[LM.rightEyeCenter] || lm[386];

  const eyes = [leftEye, rightEye];
  const time = performance.now() * 0.005;

  eyes.forEach((eye, index) => {
    canvasCtx.save();
    canvasCtx.translate(eye.x, eye.y);

    const glowRadius = 20 + Math.sin(time * 2 + index) * 6;

    // Outer Neon Glow
    const grad = canvasCtx.createRadialGradient(0, 0, 2, 0, 0, glowRadius);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(0.3, '#06b6d4');
    grad.addColorStop(0.7, '#ec4899');
    grad.addColorStop(1, 'rgba(99, 102, 241, 0)');

    canvasCtx.fillStyle = grad;
    canvasCtx.beginPath();
    canvasCtx.arc(0, 0, glowRadius, 0, Math.PI * 2);
    canvasCtx.fill();

    // Laser Spikes / Electric Beams
    canvasCtx.strokeStyle = '#06b6d4';
    canvasCtx.lineWidth = 2.5;
    for (let i = 0; i < 4; i++) {
      const rayAngle = time + (i * Math.PI / 2);
      const len = glowRadius * 1.8;
      canvasCtx.beginPath();
      canvasCtx.moveTo(0, 0);
      canvasCtx.lineTo(Math.cos(rayAngle) * len, Math.sin(rayAngle) * len);
      canvasCtx.stroke();
    }

    // Core hot point
    canvasCtx.fillStyle = '#ffffff';
    canvasCtx.beginPath();
    canvasCtx.arc(0, 0, 4, 0, Math.PI * 2);
    canvasCtx.fill();

    canvasCtx.restore();
  });
}

// -------------------------------------------------------------
// FILTER 7: 3D Futuristic Face Mesh Wireframe
// -------------------------------------------------------------
function draw3DFaceMesh(rawLandmarks, w, h) {
  canvasCtx.save();
  canvasCtx.strokeStyle = 'rgba(6, 182, 212, 0.45)';
  canvasCtx.lineWidth = 1;

  // Use MediaPipe standard FACEMESH_TESSELATION if available
  if (window.FACEMESH_TESSELATION) {
    for (const [i1, i2] of FACEMESH_TESSELATION) {
      const p1 = rawLandmarks[i1];
      const p2 = rawLandmarks[i2];
      if (p1 && p2) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(p1.x * w, p1.y * h);
        canvasCtx.lineTo(p2.x * w, p2.y * h);
        canvasCtx.stroke();
      }
    }
  } else {
    // Custom connecting mesh grid fallback
    for (let i = 0; i < rawLandmarks.length - 2; i += 3) {
      const p1 = rawLandmarks[i];
      const p2 = rawLandmarks[i + 1];
      const p3 = rawLandmarks[i + 2];
      canvasCtx.beginPath();
      canvasCtx.moveTo(p1.x * w, p1.y * h);
      canvasCtx.lineTo(p2.x * w, p2.y * h);
      canvasCtx.lineTo(p3.x * w, p3.y * h);
      canvasCtx.closePath();
      canvasCtx.stroke();
    }
  }

  // Draw glowing nodes on key feature landmarks
  canvasCtx.fillStyle = '#a855f7';
  canvasCtx.shadowColor = '#06b6d4';
  canvasCtx.shadowBlur = 8;
  for (let i = 0; i < rawLandmarks.length; i += 7) {
    const p = rawLandmarks[i];
    canvasCtx.beginPath();
    canvasCtx.arc(p.x * w, p.y * h, 1.8, 0, Math.PI * 2);
    canvasCtx.fill();
  }

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 8: Comic Cartoon Cel-Shading Filter
// -------------------------------------------------------------
function drawCartoonVideo(image, w, h) {
  // First draw base frame
  canvasCtx.drawImage(image, 0, 0, w, h);

  // Apply high contrast edge detection & saturation
  canvasCtx.save();
  canvasCtx.filter = 'contrast(170%) saturate(160%) brightness(105%)';
  canvasCtx.drawImage(canvasElement, 0, 0, w, h);
  canvasCtx.filter = 'none';

  // Comic book halftone / ink outline vignette
  const grad = canvasCtx.createRadialGradient(w / 2, h / 2, w * 0.35, w / 2, h / 2, w * 0.7);
  grad.addColorStop(0, 'rgba(0,0,0,0)');
  grad.addColorStop(1, 'rgba(0,0,0,0.65)');
  canvasCtx.fillStyle = grad;
  canvasCtx.fillRect(0, 0, w, h);

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 9: Retro VHS Glitch FX
// -------------------------------------------------------------
function drawVhsGlitch(w, h) {
  canvasCtx.save();

  // Horizontal CRT scanlines
  canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.22)';
  for (let y = 0; y < h; y += 4) {
    canvasCtx.fillRect(0, y, w, 2);
  }

  // Random glitch color slice
  if (Math.random() > 0.6) {
    const sliceY = Math.random() * h;
    const sliceH = 10 + Math.random() * 25;
    const shift = (Math.random() - 0.5) * 20;
    canvasCtx.drawImage(canvasElement, 0, sliceY, w, sliceH, shift, sliceY, w, sliceH);
  }

  // VHS On-Screen Display Text
  const date = new Date();
  const timeStr = date.toTimeString().split(' ')[0];
  canvasCtx.font = '22px "JetBrains Mono", monospace';
  canvasCtx.fillStyle = '#22c55e';
  canvasCtx.shadowColor = '#22c55e';
  canvasCtx.shadowBlur = 8;
  canvasCtx.fillText('REC [●] PLAY', 30, 45);
  canvasCtx.fillText(`OCT 14 1989  ${timeStr}`, 30, h - 30);
  canvasCtx.fillText('SP  - -:--:--', w - 210, h - 30);

  canvasCtx.restore();
}

// -------------------------------------------------------------
// FILTER 10: Alien Morph / Big Eyes Distortion
// -------------------------------------------------------------
function drawAlienMorph(image, lm, w, h) {
  const leftEye = lm[LM.leftEyeCenter] || lm[159];
  const rightEye = lm[LM.rightEyeCenter] || lm[386];
  const forehead = lm[LM.foreheadTop];

  // Draw Cute Alien Antennas above forehead
  canvasCtx.save();
  const headWidth = Math.hypot(lm[LM.rightEar].x - lm[LM.leftEar].x, lm[LM.rightEar].y - lm[LM.leftEar].y);
  [-1, 1].forEach(side => {
    canvasCtx.save();
    canvasCtx.translate(forehead.x + side * headWidth * 0.25, forehead.y - headWidth * 0.05);

    canvasCtx.strokeStyle = '#22c55e';
    canvasCtx.lineWidth = 4;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, 0);
    canvasCtx.quadraticCurveTo(side * 20, -headWidth * 0.35, side * 35, -headWidth * 0.45);
    canvasCtx.stroke();

    // Glowing green bulb at tip
    canvasCtx.beginPath();
    canvasCtx.arc(side * 35, -headWidth * 0.45, 12, 0, Math.PI * 2);
    canvasCtx.fillStyle = '#86efac';
    canvasCtx.shadowColor = '#22c55e';
    canvasCtx.shadowBlur = 15;
    canvasCtx.fill();
    canvasCtx.restore();
  });
  canvasCtx.restore();

  // Magnify eyes (Big Alien Eyes)
  [leftEye, rightEye].forEach(eye => {
    const eyeRadius = 45;
    const zoom = 1.6;

    canvasCtx.save();
    canvasCtx.beginPath();
    canvasCtx.arc(eye.x, eye.y, eyeRadius, 0, Math.PI * 2);
    canvasCtx.clip();

    // Re-draw magnified image centered on eye
    const srcX = eye.x - (eyeRadius / zoom);
    const srcY = eye.y - (eyeRadius / zoom);
    const srcW = (eyeRadius * 2) / zoom;
    const srcH = (eyeRadius * 2) / zoom;

    canvasCtx.drawImage(
      image,
      srcX, srcY, srcW, srcH,
      eye.x - eyeRadius, eye.y - eyeRadius, eyeRadius * 2, eyeRadius * 2
    );

    // Glowing green/cyan alien iris rim
    canvasCtx.strokeStyle = 'rgba(34, 197, 94, 0.6)';
    canvasCtx.lineWidth = 3;
    canvasCtx.stroke();

    canvasCtx.restore();
  });
}

// -------------------------------------------------------------
// FILTER 11: Photo Snapshot Studio & Download
// -------------------------------------------------------------
function triggerSnapshot() {
  if (isSnapping) return;
  isSnapping = true;

  let count = 3;
  countdownOverlay.textContent = count;
  countdownOverlay.classList.add('show');

  const timer = setInterval(() => {
    count--;
    if (count > 0) {
      countdownOverlay.textContent = count;
    } else {
      clearInterval(timer);
      countdownOverlay.classList.remove('show');

      // Flash Screen
      flashOverlay.classList.add('active');
      setTimeout(() => flashOverlay.classList.remove('active'), 250);

      // Capture Image from Canvas
      captureAndDownload();
      isSnapping = false;
    }
  }, 800);
}

function captureAndDownload() {
  const dataUrl = canvasElement.toDataURL('image/png');

  // Trigger browser download
  const link = document.createElement('a');
  link.download = `VisionAR_Snap_${Date.now()}.png`;
  link.href = dataUrl;
  link.click();

  // Add to thumbnail gallery
  const thumb = document.createElement('img');
  thumb.src = dataUrl;
  thumb.className = 'preview-thumb';
  thumb.title = 'Click to open';
  thumb.onclick = () => window.open(dataUrl);

  const emptyMsg = snapshotGallery.querySelector('.preview-thumb-empty');
  if (emptyMsg) emptyMsg.remove();

  snapshotGallery.prepend(thumb);
  if (snapshotGallery.children.length > 3) {
    snapshotGallery.removeChild(snapshotGallery.lastChild);
  }
}

// -------------------------------------------------------------
// Helper Drawing Utilities
// -------------------------------------------------------------
function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

// -------------------------------------------------------------
// User Interactions & UI Event Listeners
// -------------------------------------------------------------
function selectFilter(filterName) {
  if (filterName === 'snapshot') {
    triggerSnapshot();
    return;
  }

  const isMulti = multiFilterCheck.checked;

  if (isMulti) {
    if (activeFilters.has(filterName)) {
      activeFilters.delete(filterName);
    } else {
      activeFilters.add(filterName);
    }
  } else {
    activeFilters.clear();
    activeFilters.add(filterName);
  }

  updateUIActiveState();
}

function updateUIActiveState() {
  filterCards.forEach(card => {
    const f = card.dataset.filter;
    if (f === 'snapshot') return;
    if (activeFilters.has(f)) {
      card.classList.add('active');
    } else {
      card.classList.remove('active');
    }
  });

  const names = Array.from(activeFilters).join(' + ') || 'None';
  activeFilterTag.textContent = `Active: ${names.toUpperCase()}`;
}

filterCards.forEach(card => {
  card.addEventListener('click', () => {
    selectFilter(card.dataset.filter);
  });
});

btnResetFilters.addEventListener('click', () => {
  activeFilters.clear();
  updateUIActiveState();
});

btnFlipCamera.addEventListener('click', () => {
  isMirrored = !isMirrored;
  canvasElement.classList.toggle('mirrored', isMirrored);
});

btnSnapshot.addEventListener('click', triggerSnapshot);

// Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();

  // 1 through 9
  const keyMap = {
    '1': 'sunglasses',
    '2': 'mustache',
    '3': 'crown',
    '4': 'puppy',
    '5': 'clown',
    '6': 'lasereyes',
    '7': 'facemesh',
    '8': 'cartoon',
    '9': 'vhs',
    '0': 'alien',
    '-': 'snapshot',
    's': 'snapshot'
  };

  if (keyMap[key]) {
    selectFilter(keyMap[key]);
  } else if (key === 'c') {
    activeFilters.clear();
    updateUIActiveState();
  } else if (key === 'm') {
    isMirrored = !isMirrored;
    canvasElement.classList.toggle('mirrored', isMirrored);
  }
});
