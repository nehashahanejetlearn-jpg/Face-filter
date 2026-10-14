"""
VisionAR - AI Face Filter Desktop Application
JetLearn AI Inventors - 11 Features Suite

Features Included:
  [1] Cool Sunglasses (Auto-rotated polarized shades)
  [2] Vintage Gentleman Mustache
  [3] Golden Royal Crown
  [4] Puppy Ears & Dog Nose
  [5] Glowing Red Clown Nose
  [6] Cyberpunk Laser Eyes
  [7] 3D Face Mesh Tessellation
  [8] Comic Cartoon Cel-Shading Filter
  [9] Retro VHS Glitch & CRT Scanlines
  [0] Alien Morph / Big Eyes
  [S] HD Photo Snapshot (Saves to current folder)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# Initialize MediaPipe Face Mesh
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
LM_LEFT_EYE = 468   # Left iris center
LM_RIGHT_EYE = 473  # Right iris center
LM_LEFT_OUTER = 33
LM_RIGHT_OUTER = 263
LM_NOSE_TIP = 4
LM_NOSE_BOTTOM = 2
LM_UPPER_LIP = 0
LM_FOREHEAD = 10
LM_CHIN = 152
LM_LEFT_CHEEK = 234
LM_RIGHT_CHEEK = 454

# Active filter state
active_filter = 1  # Default to sunglasses
multi_mode = False
active_filters_set = {1}
flash_frames = 0

def overlay_transparent(background, overlay, x, y):
    """Safely blend an RGBA image onto a BGR background image."""
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape

    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    # Crop bounds
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

# -------------------------------------------------------------
# FILTER 1: Sunglasses / Cool Shades
# -------------------------------------------------------------
def apply_sunglasses(img, lm):
    left_eye = lm[LM_LEFT_EYE]
    right_eye = lm[LM_RIGHT_EYE]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    eye_dist = int(math.hypot(dx, dy))
    angle = math.degrees(math.atan2(dy, dx))
    center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

    w = int(eye_dist * 2.3)
    h = int(eye_dist * 0.9)
    if w <= 0 or h <= 0:
        return img

    # Draw procedural sunglasses on transparent canvas
    overlay = np.zeros((h * 3, w * 3, 4), dtype=np.uint8)
    cx, cy = int(w * 1.5), int(h * 1.5)

    lens_w = int(w * 0.44)
    bridge = int(w * 0.12)

    # Frame & Lenses
    for side in (-1, 1):
        lx = cx + (side * (bridge // 2 + lens_w // 2))
        cv2.rectangle(overlay, (lx - lens_w // 2, cy - h // 2), (lx + lens_w // 2, cy + h // 2), (20, 20, 20, 250), -1)
        cv2.rectangle(overlay, (lx - lens_w // 2, cy - h // 2), (lx + lens_w // 2, cy + h // 2), (220, 200, 0, 255), 3)
        # Specular white reflection line
        cv2.line(overlay, (lx - lens_w // 4, cy - h // 3), (lx + lens_w // 6, cy + h // 3), (255, 255, 255, 160), 3)

    # Bridge
    cv2.rectangle(overlay, (cx - bridge // 2, cy - h // 4), (cx + bridge // 2, cy - h // 8), (20, 20, 20, 255), -1)

    # Rotate overlay
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    rotated = cv2.warpAffine(overlay, M, (w * 3, h * 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return overlay_transparent(img, rotated, center[0] - cx, center[1] - cy)

# -------------------------------------------------------------
# FILTER 2: Vintage Mustache
# -------------------------------------------------------------
def apply_mustache(img, lm):
    nose_bot = lm[LM_NOSE_BOTTOM]
    lip = lm[LM_UPPER_LIP]
    left_c = lm[LM_LEFT_CHEEK]
    right_c = lm[LM_RIGHT_CHEEK]

    dx = right_c[0] - left_c[0]
    dy = right_c[1] - left_c[1]
    angle = math.degrees(math.atan2(dy, dx))
    face_w = int(math.hypot(dx, dy))

    cx = int((nose_bot[0] + lip[0]) / 2)
    cy = int((nose_bot[1] + lip[1]) / 2)

    mw = int(face_w * 0.42)
    mh = int(mw * 0.45)
    if mw <= 0 or mh <= 0:
        return img

    overlay = np.zeros((mh * 3, mw * 3, 4), dtype=np.uint8)
    ocx, ocy = int(mw * 1.5), int(mh * 1.5)

    # Mustache curves using filled ellipses
    cv2.ellipse(overlay, (ocx - int(mw * 0.22), ocy), (int(mw * 0.25), int(mh * 0.3)), -18, 0, 360, (25, 25, 25, 255), -1)
    cv2.ellipse(overlay, (ocx + int(mw * 0.22), ocy), (int(mw * 0.25), int(mh * 0.3)), 18, 0, 360, (25, 25, 25, 255), -1)
    # Curls at tips
    cv2.circle(overlay, (ocx - int(mw * 0.45), ocy - int(mh * 0.15)), int(mh * 0.18), (25, 25, 25, 255), -1)
    cv2.circle(overlay, (ocx + int(mw * 0.45), ocy - int(mh * 0.15)), int(mh * 0.18), (25, 25, 25, 255), -1)

    M = cv2.getRotationMatrix2D((ocx, ocy), -angle, 1.0)
    rotated = cv2.warpAffine(overlay, M, (mw * 3, mh * 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return overlay_transparent(img, rotated, cx - ocx, cy - ocy)

# -------------------------------------------------------------
# FILTER 3: Royal Golden Crown
# -------------------------------------------------------------
def apply_crown(img, lm):
    forehead = lm[LM_FOREHEAD]
    chin = lm[LM_CHIN]
    dx = lm[LM_RIGHT_CHEEK][0] - lm[LM_LEFT_CHEEK][0]
    dy = lm[LM_RIGHT_CHEEK][1] - lm[LM_LEFT_CHEEK][1]
    angle = math.degrees(math.atan2(dy, dx))
    face_h = int(math.hypot(chin[0] - forehead[0], chin[1] - forehead[1]))

    cw = int(face_h * 0.95)
    ch = int(cw * 0.65)
    if cw <= 0 or ch <= 0:
        return img

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

    cv2.fillPoly(overlay, [pts], (15, 185, 245, 240))  # Golden BGR
    cv2.polylines(overlay, [pts], True, (0, 120, 200, 255), 4)

    # Red/Cyan Jewels
    for gx, gy in [(ocx - int(half * 0.9), ocy - int(ch * 0.7)), (ocx, ocy - ch), (ocx + int(half * 0.9), ocy - int(ch * 0.7))]:
        cv2.circle(overlay, (gx, gy), int(cw * 0.05), (50, 50, 230, 255), -1)
        cv2.circle(overlay, (gx, gy), int(cw * 0.05), (255, 255, 255, 255), 2)

    M = cv2.getRotationMatrix2D((ocx, ocy), -angle, 1.0)
    rotated = cv2.warpAffine(overlay, M, (cw * 3, ch * 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return overlay_transparent(img, rotated, forehead[0] - ocx, forehead[1] - ocy - int(ch * 0.35))

# -------------------------------------------------------------
# FILTER 4: Puppy Ears & Nose
# -------------------------------------------------------------
def apply_puppy(img, lm):
    nose = lm[LM_NOSE_TIP]
    forehead = lm[LM_FOREHEAD]
    dx = lm[LM_RIGHT_OUTER][0] - lm[LM_LEFT_OUTER][0]
    dy = lm[LM_RIGHT_OUTER][1] - lm[LM_LEFT_OUTER][1]
    head_w = int(math.hypot(dx, dy) * 1.8)

    # Draw ears
    for side in (-1, 1):
        ear_x = int(forehead[0] + (side * head_w * 0.45))
        ear_y = int(forehead[1] - head_w * 0.2)
        ear_w = int(head_w * 0.28)
        ear_h = int(head_w * 0.55)
        # Outer brown
        cv2.ellipse(img, (ear_x, ear_y), (ear_w // 2, ear_h // 2), side * 20, 0, 360, (20, 60, 120), -1)
        # Inner pink
        cv2.ellipse(img, (ear_x, ear_y), (int(ear_w * 0.28), int(ear_h * 0.35)), side * 20, 0, 360, (180, 130, 240), -1)

    # Black Dog Nose
    nr = int(head_w * 0.12)
    cv2.ellipse(img, (nose[0], nose[1]), (int(nr * 1.2), int(nr * 0.8)), 0, 0, 360, (20, 20, 20), -1)
    cv2.circle(img, (nose[0] - int(nr * 0.3), nose[1] - int(nr * 0.2)), int(nr * 0.2), (255, 255, 255), -1)
    return img

# -------------------------------------------------------------
# FILTER 5: Glowing Red Clown Nose
# -------------------------------------------------------------
def apply_clown_nose(img, lm):
    nose = lm[LM_NOSE_TIP]
    radius = int(math.hypot(lm[LM_NOSE_BOTTOM][0] - lm[168][0], lm[LM_NOSE_BOTTOM][1] - lm[168][1]) * 0.45)
    radius = max(radius, 15)

    # Red base
    cv2.circle(img, (nose[0], nose[1]), radius, (20, 20, 230), -1)
    # Highlight
    cv2.circle(img, (nose[0] - int(radius * 0.3), nose[1] - int(radius * 0.3)), int(radius * 0.25), (255, 255, 255), -1)
    return img

# -------------------------------------------------------------
# FILTER 6: Cyberpunk Laser Eyes
# -------------------------------------------------------------
def apply_laser_eyes(img, lm):
    for eye_idx in (LM_LEFT_EYE, LM_RIGHT_EYE):
        ex, ey = lm[eye_idx]
        t = time.time() * 10
        radius = int(18 + math.sin(t) * 4)

        # Concentric glowing rings
        cv2.circle(img, (ex, ey), radius + 8, (255, 100, 0), 2)
        cv2.circle(img, (ex, ey), radius, (255, 255, 0), -1)
        cv2.circle(img, (ex, ey), int(radius * 0.4), (255, 255, 255), -1)

        # Spikes
        for angle in (0, 90, 180, 270):
            rad = math.radians(angle + t * 5)
            x2 = int(ex + math.cos(rad) * radius * 1.8)
            y2 = int(ey + math.sin(rad) * radius * 1.8)
            cv2.line(img, (ex, ey), (x2, y2), (0, 255, 255), 2)
    return img

# -------------------------------------------------------------
# FILTER 7: 3D Face Mesh Wireframe
# -------------------------------------------------------------
def apply_face_mesh(img, raw_landmarks):
    h, w, _ = img.shape
    # Draw MediaPipe Tessellation
    mp_drawing.draw_landmarks(
        image=img,
        landmark_list=raw_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    return img

# -------------------------------------------------------------
# FILTER 8: Comic Cartoon Cel Filter
# -------------------------------------------------------------
def apply_cartoon(img):
    # Bilateral smoothing + adaptive edge mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)

    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# -------------------------------------------------------------
# FILTER 9: Retro VHS Glitch FX
# -------------------------------------------------------------
def apply_vhs_glitch(img):
    h, w, _ = img.shape
    vhs = img.copy()

    # Scanlines
    vhs[0::4, :] = (vhs[0::4, :] * 0.65).astype(np.uint8)

    # Chromatic aberration / Color Channel Offset
    vhs[:, 6:, 0] = img[:, :-6, 0]  # Shift Blue
    vhs[:, :-6, 2] = img[:, 6:, 2]  # Shift Red

    # VHS Text Overlay
    timestamp = time.strftime("REC [●] OCT 14 1989  %H:%M:%S")
    cv2.putText(vhs, timestamp, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(vhs, "SP PLAY --:--:--", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vhs

# -------------------------------------------------------------
# FILTER 10: Alien Morph / Big Eyes
# -------------------------------------------------------------
def apply_alien_morph(img, lm):
    h, w, _ = img.shape
    out = img.copy()
    forehead = lm[LM_FOREHEAD]

    # Antennas
    for side in (-1, 1):
        ax = forehead[0] + side * 45
        ay = forehead[1] - 10
        cv2.line(out, (ax, ay), (ax + side * 25, ay - 60), (0, 255, 0), 4)
        cv2.circle(out, (ax + side * 25, ay - 60), 12, (100, 255, 100), -1)

    # Eye Magnification
    for eye_idx in (LM_LEFT_EYE, LM_RIGHT_EYE):
        ex, ey = lm[eye_idx]
        r = 40
        x1, y1 = max(0, ex - r), max(0, ey - r)
        x2, y2 = min(w, ex + r), min(h, ey + r)

        patch = img[y1:y2, x1:x2]
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            zoom_patch = cv2.resize(patch, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
            # Circular mask
            mask = np.zeros(patch.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (patch.shape[1] // 2, patch.shape[0] // 2), min(patch.shape[:2]) // 2, 255, -1)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(out[y1:y2, x1:x2], out[y1:y2, x1:x2], mask=mask_inv)
            fg = cv2.bitwise_and(zoom_patch, zoom_patch, mask=mask)
            out[y1:y2, x1:x2] = cv2.add(bg, fg)
            cv2.circle(out, (ex, ey), r, (0, 255, 0), 2)

    return out

# -------------------------------------------------------------
# Main Application Loop
# -------------------------------------------------------------
def main():
    global active_filter, multi_mode, active_filters_set, flash_frames
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 60)
    print(" VisionAR - 11 Features AI Face Filter Studio Active!")
    print("=" * 60)
    print(" Hotkeys:")
    print("  [1] Sunglasses     [2] Mustache       [3] Royal Crown")
    print("  [4] Puppy Dog      [5] Clown Nose     [6] Laser Eyes")
    print("  [7] 3D Face Mesh   [8] Comic Cartoon  [9] VHS Glitch")
    print("  [0] Alien Morph    [M] Multi-Stack    [S] Save Snapshot")
    print("  [C] Clear Filters  [Q] Quit")
    print("=" * 60)

    filter_names = {
        1: "1: Sunglasses", 2: "2: Mustache", 3: "3: Crown",
        4: "4: Puppy", 5: "5: Clown Nose", 6: "6: Laser Eyes",
        7: "7: 3D FaceMesh", 8: "8: Cartoon", 9: "9: VHS Glitch", 0: "0: Alien Morph"
    }

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror selfie view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Global screen filters before landmarks (Cartoon)
        if (multi_mode and 8 in active_filters_set) or (not multi_mode and active_filter == 8):
            frame = apply_cartoon(frame)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = {i: (int(p.x * w), int(p.y * h)) for i, p in enumerate(face_landmarks.landmark)}

                def should_run(fid):
                    return (multi_mode and fid in active_filters_set) or (not multi_mode and active_filter == fid)

                if should_run(7):
                    frame = apply_face_mesh(frame, face_landmarks)
                if should_run(10) or should_run(0):
                    frame = apply_alien_morph(frame, lm)
                if should_run(1):
                    frame = apply_sunglasses(frame, lm)
                if should_run(2):
                    frame = apply_mustache(frame, lm)
                if should_run(3):
                    frame = apply_crown(frame, lm)
                if should_run(4):
                    frame = apply_puppy(frame, lm)
                if should_run(5):
                    frame = apply_clown_nose(frame, lm)
                if should_run(6):
                    frame = apply_laser_eyes(frame, lm)

        # Global post-filters (VHS)
        if (multi_mode and 9 in active_filters_set) or (not multi_mode and active_filter == 9):
            frame = apply_vhs_glitch(frame)

        # FPS Calculation
        curr_time = time.time()
        fps = int(1 / max(curr_time - prev_time, 0.001))
        prev_time = curr_time

        # HUD Overlay Banner
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 20), -1)
        cv2.line(frame, (0, 50), (w, 50), (99, 102, 241), 2)

        status_str = f"FPS: {fps} | Mode: {'MULTI' if multi_mode else 'SINGLE'} | Active: "
        if multi_mode:
            status_str += ", ".join([filter_names.get(f, str(f)) for f in sorted(active_filters_set)]) or "None"
        else:
            status_str += filter_names.get(active_filter, "None")

        cv2.putText(frame, status_str, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        # Flash visual when taking snapshot
        if flash_frames > 0:
            frame[:, :] = (255, 255, 255)
            flash_frames -= 1

        cv2.imshow("VisionAR - 11 Features Face Filter Studio", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord(str(i)) for i in range(10)]:
            digit = int(chr(key))
            if multi_mode:
                if digit in active_filters_set:
                    active_filters_set.remove(digit)
                else:
                    active_filters_set.add(digit)
            else:
                active_filter = digit
        elif key == ord('m'):
            multi_mode = not multi_mode
            if multi_mode:
                active_filters_set = {active_filter}
        elif key == ord('c'):
            active_filters_set.clear()
            active_filter = -1
        elif key == ord('s'):
            # Save snapshot
            flash_frames = 2
            snap_filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_filename, frame)
            print(f"[SNAPSHOT SAVED] -> {snap_filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
