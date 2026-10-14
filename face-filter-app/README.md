# 🌟 VisionAR - 11-Feature AI Face Filter Studio

Real-time 468-point neural face tracking filter studio built with Google MediaPipe.

---

## 🚀 Features Included (11 Total)

| # | Feature / Filter | Description |
|---|---|---|
| **1** | 🕶️ **Cool Sunglasses** | Polarized dark shades with realistic reflection and dynamic angle tracking |
| **2** | 🥸 **Vintage Mustache** | Classic curved handlebar mustache anchored to upper lip |
| **3** | 👑 **Royal Crown** | Golden monarch crown with sparkling red & cyan gemstones |
| **4** | 🐶 **Puppy Dog** | Floppy puppy ears on temples + cute black dog nose |
| **5** | 🔴 **Clown Nose** | Glowing 3D red rubber clown nose with shiny specular highlight |
| **6** | ⚡ **Cyberpunk Laser Eyes** | Glowing pupil rings with pulsing electric laser beams |
| **7** | 🕸️ **3D Face Mesh Wireframe** | 468-point neural tessellation grid with cyan cyber glow |
| **8** | 🎨 **Comic Cartoon Filter** | Cel-shaded pop art filter with edge detection |
| **9** | 📼 **Retro VHS Glitch** | CRT horizontal scanlines, RGB chromatic split, and 1989 timestamp |
| **10** | 👽 **Alien Big Eyes Morph** | Extraterrestrial eye magnification and glowing green antennas |
| **11** | 📸 **Photo Studio (Snapshot)** | 3-second countdown, screen flash, and HD photo download |

---

## 🌐 Option 1: Web Application (Browser)

**Zero installation required!**

1. Navigate to: `face-filter-app/web/index.html`
2. Double-click [index.html](file:///c:/Users/ajupi/Documents/JetLearn/AI-Simulation-Lab-main/face-filter-app/web/index.html) to open in Chrome, Edge, or Firefox.
3. Allow Camera permissions when prompted.
4. Click any filter card or use your keyboard:
   - Keys **[1]** through **[9]**, **[0]** to select filters
   - Key **[S]** or **[-]** to take a snapshot
   - Key **[C]** to clear filters
   - Key **[M]** to mirror/flip camera
   - Toggle **Multi-Filter Stack** to wear multiple filters at the same time (e.g., Crown + Shades + Mustache)!

---

## 🐍 Option 2: Python Desktop Application (OpenCV + MediaPipe)

1. Open your terminal or Command Prompt in `face-filter-app/python/`:
   ```bash
   cd "face-filter-app/python"
   pip install -r requirements.txt
   python face_filter.py
   ```
   *(Or simply double-click `run_app.bat`!)*

2. **Keyboard Controls inside the OpenCV window:**
   - Keys `1` to `9`, `0`: Switch filters
   - Key `m`: Toggle Multi-Filter stack mode
   - Key `s`: Take and save an HD screenshot to disk
   - Key `c`: Clear filters
   - Key `q`: Quit application
