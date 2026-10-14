"""
VisionAR - Flask Web Localhost App (Python + OpenCV + MediaPipe)
Streams real-time 11 face filters directly to http://localhost:5000 in your browser!
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import webbrowser
from flask import Flask, Response, render_template_string, jsonify, request

app = Flask(__name__)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices
LM_LEFT_EYE = 468
LM_RIGHT_EYE = 473
LM_LEFT_OUTER = 33
LM_RIGHT_OUTER = 263
LM_NOSE_TIP = 4
LM_NOSE_BOTTOM = 2
LM_UPPER_LIP = 0
LM_FOREHEAD = 10
LM_CHIN = 152
LM_LEFT_CHEEK = 234
LM_RIGHT_CHEEK = 454

# Global State
active_filter = 1
multi_mode = False
active_filters_set = {1}
flash_frames = 0
camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape

    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ol_w, bg_w), min(y + ol_h, bg_h)

    ol_x1, ol_y1 = max(0, -x), max(0, -y)
    ol_x2, ol_y2 = ol_x1 + (x2 - x1), ol_y1 + (y2 - y1)

    alpha = overlay[ol_y1:ol_y2, ol_x1:ol_x2, 3] / 255.0
    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            alpha * overlay[ol_y1:ol_y2, ol_x1:ol_x2, c] +
            (1.0 - alpha) * background[y1:y2, x1:x2, c]
        )
    return background

# Filter Renderers
def apply_sunglasses(img, lm):
    left_eye, right_eye = lm[LM_LEFT_EYE], lm[LM_RIGHT_EYE]
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    eye_dist = int(math.hypot(dx, dy))
    angle = math.degrees(math.atan2(dy, dx))
    center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

    w, h = int(eye_dist * 2.3), int(eye_dist * 0.9)
    if w <= 0 or h <= 0: return img

    overlay = np.zeros((h * 3, w * 3, 4), dtype=np.uint8)
    cx, cy = int(w * 1.5), int(h * 1.5)
    lens_w, bridge = int(w * 0.44), int(w * 0.12)

    for side in (-1, 1):
        lx = cx + (side * (bridge // 2 + lens_w // 2))
        cv2.rectangle(overlay, (lx - lens_w // 2, cy - h // 2), (lx + lens_w // 2, cy + h // 2), (20, 20, 20, 250), -1)
        cv2.rectangle(overlay, (lx - lens_w // 2, cy - h // 2), (lx + lens_w // 2, cy + h // 2), (220, 200, 0, 255), 3)
        cv2.line(overlay, (lx - lens_w // 4, cy - h // 3), (lx + lens_w // 6, cy + h // 3), (255, 255, 255, 160), 3)

    cv2.rectangle(overlay, (cx - bridge // 2, cy - h // 4), (cx + bridge // 2, cy - h // 8), (20, 20, 20, 255), -1)

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    rotated = cv2.warpAffine(overlay, M, (w * 3, h * 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return overlay_transparent(img, rotated, center[0] - cx, center[1] - cy)

def apply_mustache(img, lm):
    nose_bot, lip = lm[LM_NOSE_BOTTOM], lm[LM_UPPER_LIP]
    left_c, right_c = lm[LM_LEFT_CHEEK], lm[LM_RIGHT_CHEEK]
    dx, dy = right_c[0] - left_c[0], right_c[1] - left_c[1]
    angle = math.degrees(math.atan2(dy, dx))
    face_w = int(math.hypot(dx, dy))

    cx = int((nose_bot[0] + lip[0]) / 2)
    cy = int((nose_bot[1] + lip[1]) / 2)
    mw, mh = int(face_w * 0.42), int(face_w * 0.42 * 0.45)
    if mw <= 0 or mh <= 0: return img

    overlay = np.zeros((mh * 3, mw * 3, 4), dtype=np.uint8)
    ocx, ocy = int(mw * 1.5), int(mh * 1.5)

    cv2.ellipse(overlay, (ocx - int(mw * 0.22), ocy), (int(mw * 0.25), int(mh * 0.3)), -18, 0, 360, (25, 25, 25, 255), -1)
    cv2.ellipse(overlay, (ocx + int(mw * 0.22), ocy), (int(mw * 0.25), int(mh * 0.3)), 18, 0, 360, (25, 25, 25, 255), -1)
    cv2.circle(overlay, (ocx - int(mw * 0.45), ocy - int(mh * 0.15)), int(mh * 0.18), (25, 25, 25, 255), -1)
    cv2.circle(overlay, (ocx + int(mw * 0.45), ocy - int(mh * 0.15)), int(mh * 0.18), (25, 25, 25, 255), -1)

    M = cv2.getRotationMatrix2D((ocx, ocy), -angle, 1.0)
    rotated = cv2.warpAffine(overlay, M, (mw * 3, mh * 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return overlay_transparent(img, rotated, cx - ocx, cy - ocy)

def apply_crown(img, lm):
    forehead, chin = lm[LM_FOREHEAD], lm[LM_CHIN]
    dx, dy = lm[LM_RIGHT_CHEEK][0] - lm[LM_LEFT_CHEEK][0], lm[LM_RIGHT_CHEEK][1] - lm[LM_LEFT_CHEEK][1]
    angle = math.degrees(math.atan2(dy, dx))
    face_h = int(math.hypot(chin[0] - forehead[0], chin[1] - forehead[1]))
    cw, ch = int(face_h * 0.95), int(face_h * 0.95 * 0.65)
    if cw <= 0 or ch <= 0: return img

    overlay = np.zeros((ch * 3, cw * 3, 4), dtype=np.uint8)
    ocx, ocy = int(cw * 1.5), int(ch * 1.5)
    half = cw // 2

    pts = np.array([
        [ocx - half, ocy],
        [ocx - int(half * 0.9), ocy - int(ch * 0.7)],
        [ocx - int(half * 0.45), ocy - int(ch * 0.4)],
        [ocx, ocy - ch],
        [ocx + int(half * 0.45), ocy - int(ch * 0.4)],
        [ocx + int(half * 0.9), ocy - int(ch * 0.7)],
        [ocx + half, ocy]
    ], np.int32)

    cv2.fillPoly(overlay, [pts], (15, 185, 245, 240))
    cv2.polylines(overlay, [pts], True, (0, 120, 200, 255), 4)

    for gx, gy in [(ocx - int(half * 0.9), ocy - int(ch * 0.7)), (ocx, ocy - ch), (ocx + int(half * 0.9), ocy - int(ch * 0.7))]:
        cv2.circle(overlay, (gx, gy), int(cw * 0.05), (50, 50, 230, 255), -1)

    M = cv2.getRotationMatrix2D((ocx, ocy), -angle, 1.0)
    rotated = cv2.warpAffine(overlay, M, (cw * 3, ch * 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return overlay_transparent(img, rotated, forehead[0] - ocx, forehead[1] - ocy - int(ch * 0.35))

def apply_puppy(img, lm):
    nose, forehead = lm[LM_NOSE_TIP], lm[LM_FOREHEAD]
    dx, dy = lm[LM_RIGHT_OUTER][0] - lm[LM_LEFT_OUTER][0], lm[LM_RIGHT_OUTER][1] - lm[LM_LEFT_OUTER][1]
    head_w = int(math.hypot(dx, dy) * 1.8)

    for side in (-1, 1):
        ear_x, ear_y = int(forehead[0] + (side * head_w * 0.45)), int(forehead[1] - head_w * 0.2)
        ear_w, ear_h = int(head_w * 0.28), int(head_w * 0.55)
        cv2.ellipse(img, (ear_x, ear_y), (ear_w // 2, ear_h // 2), side * 20, 0, 360, (20, 60, 120), -1)
        cv2.ellipse(img, (ear_x, ear_y), (int(ear_w * 0.28), int(ear_h * 0.35)), side * 20, 0, 360, (180, 130, 240), -1)

    nr = int(head_w * 0.12)
    cv2.ellipse(img, (nose[0], nose[1]), (int(nr * 1.2), int(nr * 0.8)), 0, 0, 360, (20, 20, 20), -1)
    cv2.circle(img, (nose[0] - int(nr * 0.3), nose[1] - int(nr * 0.2)), int(nr * 0.2), (255, 255, 255), -1)
    return img

def apply_clown_nose(img, lm):
    nose = lm[LM_NOSE_TIP]
    radius = max(int(math.hypot(lm[LM_NOSE_BOTTOM][0] - lm[168][0], lm[LM_NOSE_BOTTOM][1] - lm[168][1]) * 0.45), 15)
    cv2.circle(img, (nose[0], nose[1]), radius, (20, 20, 230), -1)
    cv2.circle(img, (nose[0] - int(radius * 0.3), nose[1] - int(radius * 0.3)), int(radius * 0.25), (255, 255, 255), -1)
    return img

def apply_laser_eyes(img, lm):
    for eye_idx in (LM_LEFT_EYE, LM_RIGHT_EYE):
        ex, ey = lm[eye_idx]
        t = time.time() * 10
        radius = int(18 + math.sin(t) * 4)
        cv2.circle(img, (ex, ey), radius + 8, (255, 100, 0), 2)
        cv2.circle(img, (ex, ey), radius, (255, 255, 0), -1)
        cv2.circle(img, (ex, ey), int(radius * 0.4), (255, 255, 255), -1)
        for angle in (0, 90, 180, 270):
            rad = math.radians(angle + t * 5)
            cv2.line(img, (ex, ey), (int(ex + math.cos(rad) * radius * 1.8), int(ey + math.sin(rad) * radius * 1.8)), (0, 255, 255), 2)
    return img

def apply_face_mesh(img, raw_landmarks):
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=raw_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    return img

def apply_cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def apply_vhs_glitch(img):
    h, w, _ = img.shape
    vhs = img.copy()
    vhs[0::4, :] = (vhs[0::4, :] * 0.65).astype(np.uint8)
    vhs[:, 6:, 0] = img[:, :-6, 0]
    vhs[:, :-6, 2] = img[:, 6:, 2]
    timestamp = time.strftime("REC [●] OCT 14 1989  %H:%M:%S")
    cv2.putText(vhs, timestamp, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return vhs

def apply_alien_morph(img, lm):
    h, w, _ = img.shape
    out = img.copy()
    forehead = lm[LM_FOREHEAD]
    for side in (-1, 1):
        ax, ay = forehead[0] + side * 45, forehead[1] - 10
        cv2.line(out, (ax, ay), (ax + side * 25, ay - 60), (0, 255, 0), 4)
        cv2.circle(out, (ax + side * 25, ay - 60), 12, (100, 255, 100), -1)

    for eye_idx in (LM_LEFT_EYE, LM_RIGHT_EYE):
        ex, ey = lm[eye_idx]
        r = 40
        x1, y1 = max(0, ex - r), max(0, ey - r)
        x2, y2 = min(w, ex + r), min(h, ey + r)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            zoom_patch = cv2.resize(patch, (patch.shape[1], patch.shape[0]))
            mask = np.zeros(patch.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (patch.shape[1] // 2, patch.shape[0] // 2), min(patch.shape[:2]) // 2, 255, -1)
            bg = cv2.bitwise_and(out[y1:y2, x1:x2], out[y1:y2, x1:x2], mask=cv2.bitwise_not(mask))
            fg = cv2.bitwise_and(zoom_patch, zoom_patch, mask=mask)
            out[y1:y2, x1:x2] = cv2.add(bg, fg)
            cv2.circle(out, (ex, ey), r, (0, 255, 0), 2)
    return out

def generate_frames():
    global flash_frames
    cap = get_camera()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if (multi_mode and 8 in active_filters_set) or (not multi_mode and active_filter == 8):
            frame = apply_cartoon(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                lm = {i: (int(p.x * w), int(p.y * h)) for i, p in enumerate(fl.landmark)}

                def should_run(fid):
                    return (multi_mode and fid in active_filters_set) or (not multi_mode and active_filter == fid)

                if should_run(7): frame = apply_face_mesh(frame, fl)
                if should_run(0): frame = apply_alien_morph(frame, lm)
                if should_run(1): frame = apply_sunglasses(frame, lm)
                if should_run(2): frame = apply_mustache(frame, lm)
                if should_run(3): frame = apply_crown(frame, lm)
                if should_run(4): frame = apply_puppy(frame, lm)
                if should_run(5): frame = apply_clown_nose(frame, lm)
                if should_run(6): frame = apply_laser_eyes(frame, lm)

        if (multi_mode and 9 in active_filters_set) or (not multi_mode and active_filter == 9):
            frame = apply_vhs_glitch(frame)

        if flash_frames > 0:
            frame[:, :] = (255, 255, 255)
            flash_frames -= 1

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# HTML Web UI Template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>VisionAR - Localhost Studio</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
  <style>
    * { margin:0; padding:0; box-sizing:border-box; font-family:'Outfit',sans-serif; }
    body { background: radial-gradient(circle at 50% 0%, #1e1b4b, #0a0c14 70%); color:#fff; padding:1.5rem; text-align:center; min-height:100vh; }
    h1 { font-size:2rem; font-weight:800; margin-bottom:0.5rem; }
    .badge { background:linear-gradient(90deg,#06b6d4,#6366f1); padding:0.2rem 0.6rem; border-radius:20px; font-size:0.75rem; vertical-align:middle; }
    .stream-box { max-width:960px; margin:1rem auto; border-radius:18px; overflow:hidden; border:2px solid rgba(255,255,255,0.1); box-shadow:0 10px 40px rgba(0,0,0,0.7); background:#000; }
    .stream-box img { width:100%; height:auto; display:block; }
    .dock { display:flex; flex-wrap:wrap; gap:0.6rem; justify-content:center; max-width:1050px; margin:1.5rem auto; }
    .btn { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); color:#fff; padding:0.7rem 1.1rem; border-radius:12px; cursor:pointer; font-weight:700; font-size:0.95rem; transition:all 0.2s; display:flex; align-items:center; gap:0.5rem; }
    .btn:hover { background:rgba(99,102,241,0.3); border-color:#6366f1; transform:translateY(-2px); }
    .btn.active { background:linear-gradient(135deg, #06b6d4, #6366f1); border-color:#06b6d4; box-shadow:0 0 15px rgba(6,182,212,0.4); }
    .snap-btn { background:linear-gradient(135deg,#ec4899,#a855f7); border:none; box-shadow:0 0 15px rgba(236,72,153,0.5); }
    .snap-btn:hover { transform:scale(1.05); }
    .status-line { margin-top:0.5rem; color:#94a3b8; font-family:'JetBrains Mono',monospace; font-size:0.85rem; }
  </style>
</head>
<body>
  <h1>VisionAR <span class="badge">Localhost Python Engine</span></h1>
  <p>Streaming OpenCV + MediaPipe directly to your browser at http://localhost:5000</p>
  
  <div class="stream-box">
    <img src="/video_feed" alt="Webcam Stream">
  </div>

  <div class="dock">
    <button class="btn active" onclick="setFilter(1, this)">🕶️ 1. Sunglasses</button>
    <button class="btn" onclick="setFilter(2, this)">🥸 2. Mustache</button>
    <button class="btn" onclick="setFilter(3, this)">👑 3. Crown</button>
    <button class="btn" onclick="setFilter(4, this)">🐶 4. Puppy</button>
    <button class="btn" onclick="setFilter(5, this)">🔴 5. Clown</button>
    <button class="btn" onclick="setFilter(6, this)">⚡ 6. Laser Eyes</button>
    <button class="btn" onclick="setFilter(7, this)">🕸️ 7. 3D Mesh</button>
    <button class="btn" onclick="setFilter(8, this)">🎨 8. Cartoon</button>
    <button class="btn" onclick="setFilter(9, this)">📼 9. VHS Glitch</button>
    <button class="btn" onclick="setFilter(0, this)">👽 0. Alien Morph</button>
    <button class="btn snap-btn" onclick="snapshot()">📸 Take Snapshot</button>
    <button class="btn" onclick="clearFilters()">❌ Clear</button>
  </div>
  <div class="status-line" id="statusLine">Active: 🕶️ Sunglasses | Press keys 1-9, 0, S on keyboard</div>

  <script>
    function setFilter(id, elem) {
      fetch('/set_filter/' + id)
        .then(r => r.json())
        .then(data => {
          document.querySelectorAll('.dock .btn').forEach(b => b.classList.remove('active'));
          if (elem) elem.classList.add('active');
          document.getElementById('statusLine').innerText = 'Active: ' + data.name;
        });
    }
    function clearFilters() {
      fetch('/clear_filters').then(() => {
        document.querySelectorAll('.dock .btn').forEach(b => b.classList.remove('active'));
        document.getElementById('statusLine').innerText = 'Active: None';
      });
    }
    function snapshot() {
      fetch('/snapshot').then(r => r.json()).then(d => {
        alert('Snapshot saved to disk: ' + d.filename);
      });
    }
    window.addEventListener('keydown', (e) => {
      const k = e.key;
      if (k >= '0' && k <= '9') {
        setFilter(parseInt(k));
      } else if (k.toLowerCase() === 's') {
        snapshot();
      } else if (k.toLowerCase() === 'c') {
        clearFilters();
      }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_filter/<int:fid>')
def set_filter(fid):
    global active_filter, multi_mode, active_filters_set
    active_filter = fid
    names = {
        1: "Sunglasses", 2: "Mustache", 3: "Crown", 4: "Puppy Dog",
        5: "Clown Nose", 6: "Laser Eyes", 7: "3D Mesh", 8: "Cartoon",
        9: "VHS Glitch", 0: "Alien Morph"
    }
    return jsonify({"status": "ok", "active": fid, "name": names.get(fid, str(fid))})

@app.route('/clear_filters')
def clear_filters():
    global active_filter
    active_filter = -1
    return jsonify({"status": "cleared"})

@app.route('/snapshot')
def snapshot():
    global flash_frames
    flash_frames = 2
    cap = get_camera()
    ret, frame = cap.read()
    filename = f"snapshot_{int(time.time())}.jpg"
    if ret:
        cv2.imwrite(filename, frame)
    return jsonify({"status": "saved", "filename": filename})

def main():
    port = 5000
    url = f"http://localhost:{port}"
    print("=" * 60)
    print(f" VisionAR Flask Web Server running at: {url}")
    print("=" * 60)
    webbrowser.open(url)
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == '__main__':
    main()
