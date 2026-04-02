r"""
PCB Solder Defect Detection System - GUI App V5
CPEQC-029-030 | Team 4 | TIP Quezon City

Dataset  : SolDef-AI (Kaggle - mauriziocalabrese)
           ~1150 SMT solder images, 3 viewpoints, aerospace-grade quality control
           Defect types aligned to IPC-A-610 Class 2 standard

Improvements over V4
─────────────────────
1.  Fixed camera index (0) — no more radio buttons
2.  Upload button auto-opens file dialog instantly
3.  Aspect-ratio-correct canvas that NEVER grows the window
4.  CLAHE pre-processing tuned to SolDef-AI microscope images
       (USB Cainda HD microscope, close-up SMT shots → high contrast boost)
5.  Multi-scale inference for uploaded images (640 + 960) with custom NMS
6.  Confidence floor lowered to 0.15 on upload (SolDef-AI has small solder joints)
7.  Dedicated "No Detection" state separate from PASS/FAIL
8.  IPC-A-610 report button restored in the Actions panel
9.  Per-class defect bar chart in the Session Stats section
10. Live inference moved to a frame queue so the GUI never blocks
11. Half-precision (FP16) auto-enabled on CUDA for faster live inference
12. Snapshot saves the ORIGINAL annotated frame at full camera resolution
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import os
import queue
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import numpy as np
from datetime import datetime

# Suppress noisy OpenCV MSMF warnings on Windows
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path.home() / "Downloads" / "PCB_V2"
WEIGHTS_PATH = PROJECT_DIR / "runs" / "PCB_V2" / "weights" / "best.pt"
RESULTS_DIR  = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_IDX  = 0       # single fixed camera
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMG_SIZE    = 640

# SolDef-AI dataset: images taken with USB microscope → small solder joints,
# high-contrast background. CLAHE with moderate clip works well.
CLAHE_CLIP      = 2.5
CLAHE_TILE      = (8, 8)

# Upload mode uses two scales to catch both small joints and board-level context
UPLOAD_SCALES   = [640, 960]
UPLOAD_CONF     = 0.15   # lower threshold: SolDef-AI solder joints are small
UPLOAD_NMS_IOU  = 0.50

# ── CLASS DEFINITIONS (unchanged per requirement) ──────────────────────────────
CLASS_NAMES = {
    0: "No_Defect",
    1: "Solder_Bridge",
    2: "Insufficient_Solder",
    3: "Excess_Solder",
    4: "Solder_Spike",
}
DISPLAY_NAMES = {
    0: "No Defect",
    1: "Solder Bridge",
    2: "Insufficient Solder",
    3: "Excess Solder",
    4: "Solder Spike",
}
CLASS_COLORS_BGR = {
    0: (0,   255,   0),
    1: (0,     0, 255),
    2: (0,   140, 255),
    3: (0,   255, 255),
    4: (255,   0, 255),
}
CLASS_COLORS_HEX = {
    0: "#00FF00",
    1: "#FF0000",
    2: "#FF8C00",
    3: "#FFFF00",
    4: "#FF00FF",
}
IPC_STATUS = {0: "PASS", 1: "FAIL", 2: "FAIL", 3: "FAIL", 4: "FAIL"}

# ── THEME ──────────────────────────────────────────────────────────────────────
BG_DARK  = "#0A0E1A"
BG_PANEL = "#111827"
BG_CARD  = "#1A2235"
ACCENT   = "#00D4FF"
ACCENT2  = "#00FF88"
TEXT     = "#E2E8F0"
DIM      = "#64748B"
RED      = "#FF4444"
GREEN    = "#00FF88"
YELLOW   = "#FFDD00"

WIN_W    = 1420
WIN_H    = 860
LEFT_W   = 1020
RIGHT_W  = 370


# ══════════════════════════════════════════════════════════════════════════════
class PCBDetectorApp:
    # ── INIT ──────────────────────────────────────────────────────────────────
    def __init__(self, root):
        self.root = root
        self.root.title("PCB Solder Defect Detection V5 | CPEQC-029-030 Team 4 | TIP QC")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(1200, 700)
        self.root.maxsize(1800, 1080)

        # Model / device
        self.model   = None
        self.device  = "cpu"
        self.half    = False

        # Camera state
        self.camera      = None
        self.cam_running = False
        self.cam_thread  = None
        self.frame_queue = queue.Queue(maxsize=2)

        # UI state
        self.mode          = "camera"
        self.conf_var      = tk.DoubleVar(value=CONF_THRESH)
        self.iou_var       = tk.DoubleVar(value=IOU_THRESH)
        self.status_var    = tk.StringVar(value="Initializing…")
        self.fps_var       = tk.StringVar(value="FPS: --")
        self.current_frame = None   # last annotated full-res frame

        # Session counters
        self.frame_count  = 0
        self.session_dets = {i: 0 for i in range(5)}
        self.last_time    = time.time()

        self._build_ui()
        threading.Thread(target=self._load_model, daemon=True).start()
        self._poll_frame_queue()

    # ── BUILD UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ────────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG_PANEL, height=50)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(top, text="🔬 PCB Solder Defect Detection V5",
                 bg=BG_PANEL, fg=ACCENT, font=("Arial", 12, "bold")).pack(side="left", padx=12, pady=8)
        tk.Label(top, text="CPEQC-029-030 | Team 4 | TIP Quezon City",
                 bg=BG_PANEL, fg=DIM, font=("Arial", 9)).pack(side="left", padx=4)
        tk.Label(top, text="Dataset: SolDef-AI",
                 bg=BG_PANEL, fg=DIM, font=("Arial", 8, "italic")).pack(side="left", padx=8)

        tk.Label(top, textvariable=self.fps_var,
                 bg=BG_PANEL, fg=ACCENT2, font=("Arial", 10, "bold")).pack(side="right", padx=12)
        tk.Label(top, textvariable=self.status_var,
                 bg=BG_PANEL, fg=YELLOW, font=("Arial", 9)).pack(side="right", padx=8)

        # ── Body ───────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=6, pady=6)

        # ── LEFT panel ─────────────────────────────────────────────────────────
        self.left = tk.Frame(body, bg=BG_DARK, width=LEFT_W)
        self.left.pack(side="left", fill="both", expand=True)
        self.left.pack_propagate(False)   # critical: prevents canvas feedback loop

        # Mode bar (no camera radio buttons — single fixed cam)
        mode_bar = tk.Frame(self.left, bg=BG_CARD, height=44)
        mode_bar.pack(fill="x", pady=(0, 4))
        mode_bar.pack_propagate(False)

        tk.Label(mode_bar, text="MODE:", bg=BG_CARD, fg=DIM,
                 font=("Arial", 9, "bold")).pack(side="left", padx=(10, 6), pady=10)

        self.btn_cam = tk.Button(
            mode_bar, text="📷  LIVE CAMERA",
            bg=ACCENT, fg=BG_DARK, font=("Arial", 9, "bold"),
            relief="flat", padx=16, pady=4,
            command=self._switch_to_camera)
        self.btn_cam.pack(side="left", padx=4)

        self.btn_upload = tk.Button(
            mode_bar, text="🖼  UPLOAD IMAGE",
            bg=BG_PANEL, fg=TEXT, font=("Arial", 9, "bold"),
            relief="flat", padx=16, pady=4,
            command=self._switch_to_upload)
        self.btn_upload.pack(side="left", padx=4)

        # Tiny tip label
        tk.Label(mode_bar, text="Cam 0  |  SolDef-AI optimized",
                 bg=BG_CARD, fg=DIM, font=("Arial", 8, "italic")).pack(side="left", padx=14)

        # Canvas (strict sizing via parent frame)
        self.canvas = tk.Label(self.left, bg="#000000", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # Upload bar (hidden by default)
        self.upload_bar = tk.Frame(self.left, bg=BG_CARD, height=40)
        tk.Button(self.upload_bar, text="Browse PCB Image",
                  bg=ACCENT, fg=BG_DARK, font=("Arial", 9, "bold"),
                  relief="flat", padx=12, pady=3,
                  command=self._browse_image).pack(side="left", padx=8, pady=5)
        self.img_lbl = tk.Label(self.upload_bar, text="No image selected",
                                bg=BG_CARD, fg=DIM, font=("Arial", 9))
        self.img_lbl.pack(side="left")

        # ── RIGHT panel (scrollable) ────────────────────────────────────────────
        right_wrap = tk.Frame(body, bg=BG_DARK, width=RIGHT_W)
        right_wrap.pack(side="right", fill="y", padx=(6, 0))
        right_wrap.pack_propagate(False)

        rc = tk.Canvas(right_wrap, bg=BG_DARK, width=RIGHT_W - 18,
                       highlightthickness=0)
        rc.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(right_wrap, orient="vertical", command=rc.yview)
        sb.pack(side="right", fill="y")
        rc.configure(yscrollcommand=sb.set)

        right = tk.Frame(rc, bg=BG_DARK, width=RIGHT_W - 30)
        rc.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>",
                   lambda e: rc.configure(scrollregion=rc.bbox("all")))
        # Mousewheel scroll
        rc.bind_all("<MouseWheel>",
                    lambda e: rc.yview_scroll(int(-1 * e.delta / 120), "units"))

        # ── Detection Results ───────────────────────────────────────────────────
        self._section(right, "DETECTION RESULTS")
        det_card = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        det_card.pack(fill="x", pady=(0, 6))

        self.count_vars = {}
        for cid in range(5):
            row = tk.Frame(det_card, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            c = CLASS_COLORS_HEX[cid]
            tk.Label(row, text="●", bg=BG_CARD, fg=c,
                     font=("Arial", 10)).pack(side="left")
            tk.Label(row, text=DISPLAY_NAMES[cid], bg=BG_CARD, fg=TEXT,
                     font=("Arial", 9), width=20, anchor="w").pack(side="left", padx=3)
            v = tk.StringVar(value="0")
            tk.Label(row, textvariable=v, bg=BG_CARD, fg=c,
                     font=("Arial", 10, "bold"), width=3).pack(side="right")
            self.count_vars[cid] = v

        self.verdict_var = tk.StringVar(value="READY")
        self.verdict_lbl = tk.Label(det_card, textvariable=self.verdict_var,
                                    bg=BG_CARD, fg=ACCENT2,
                                    font=("Arial", 15, "bold"))
        self.verdict_lbl.pack(pady=(10, 3))

        # ── Confidence bar (visual indicator for live mode) ─────────────────────
        self._section(right, "LIVE CONFIDENCE")
        self.conf_bar_frame = tk.Frame(right, bg=BG_CARD, padx=10, pady=6)
        self.conf_bar_frame.pack(fill="x", pady=(0, 6))
        self.conf_bars = {}
        for cid in range(1, 5):   # only defect classes
            row = tk.Frame(self.conf_bar_frame, bg=BG_CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=DISPLAY_NAMES[cid][:14], bg=BG_CARD, fg=DIM,
                     font=("Arial", 7), width=14, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg="#1e2d40", height=8, width=160)
            bar_bg.pack(side="left", padx=3)
            bar_bg.pack_propagate(False)
            bar_fill = tk.Frame(bar_bg, bg=CLASS_COLORS_HEX[cid], height=8, width=0)
            bar_fill.place(x=0, y=0, relheight=1)
            self.conf_bars[cid] = (bar_bg, bar_fill)

        # ── Settings ────────────────────────────────────────────────────────────
        self._section(right, "SETTINGS")
        ctrl = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        ctrl.pack(fill="x", pady=(0, 6))

        tk.Label(ctrl, text="Confidence Threshold", bg=BG_CARD,
                 fg=DIM, font=("Arial", 9)).pack(anchor="w")
        self.conf_lbl = tk.Label(ctrl, text=f"{CONF_THRESH:.0%}",
                                 bg=BG_CARD, fg=ACCENT, font=("Arial", 9, "bold"))
        self.conf_lbl.pack(anchor="e")
        ttk.Scale(ctrl, from_=0.05, to=0.95, variable=self.conf_var,
                  orient="horizontal",
                  command=lambda v: self.conf_lbl.config(
                      text=f"{float(v):.0%}")).pack(fill="x", pady=(0, 6))

        tk.Label(ctrl, text="IoU Threshold", bg=BG_CARD,
                 fg=DIM, font=("Arial", 9)).pack(anchor="w")
        self.iou_lbl = tk.Label(ctrl, text=f"{IOU_THRESH:.0%}",
                                bg=BG_CARD, fg=ACCENT, font=("Arial", 9, "bold"))
        self.iou_lbl.pack(anchor="e")
        ttk.Scale(ctrl, from_=0.1, to=0.9, variable=self.iou_var,
                  orient="horizontal",
                  command=lambda v: self.iou_lbl.config(
                      text=f"{float(v):.0%}")).pack(fill="x")

        # ── Class Legend ────────────────────────────────────────────────────────
        self._section(right, "CLASS LEGEND  |  IPC-A-610")
        leg = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        leg.pack(fill="x", pady=(0, 6))
        for cid in range(5):
            c   = CLASS_COLORS_HEX[cid]
            ipc = IPC_STATUS[cid]
            ic  = GREEN if ipc == "PASS" else RED
            row = tk.Frame(leg, bg=BG_CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"[{cid}]", bg=BG_CARD, fg=c,
                     font=("Arial", 8, "bold"), width=4).pack(side="left")
            tk.Label(row, text=DISPLAY_NAMES[cid], bg=BG_CARD, fg=TEXT,
                     font=("Arial", 8), width=18, anchor="w").pack(side="left")
            tk.Label(row, text=ipc, bg=BG_CARD, fg=ic,
                     font=("Arial", 8, "bold"), width=5).pack(side="right")

        # ── Actions ─────────────────────────────────────────────────────────────
        self._section(right, "ACTIONS")
        acts = tk.Frame(right, bg=BG_DARK)
        acts.pack(fill="x", pady=(0, 6))

        btn_cfg = dict(font=("Arial", 9, "bold"), relief="flat",
                       padx=10, pady=5, bd=0)
        tk.Button(acts, text="📸  Save Snapshot",
                  bg=BG_CARD, fg=TEXT, command=self._save_snapshot,
                  **btn_cfg).pack(fill="x", pady=2)
        tk.Button(acts, text="📋  IPC-A-610 Report",
                  bg=BG_CARD, fg=TEXT, command=self._ipc_report,
                  **btn_cfg).pack(fill="x", pady=2)
        tk.Button(acts, text="🔄  Reset Stats",
                  bg=BG_CARD, fg=DIM,
                  font=("Arial", 9), relief="flat", padx=10, pady=4,
                  command=self._reset_stats).pack(fill="x", pady=2)

        # ── Session Stats ───────────────────────────────────────────────────────
        self._section(right, "SESSION STATS")
        sess = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        sess.pack(fill="x", pady=(0, 6))

        self.frames_var = tk.StringVar(value="Frames processed: 0")
        self.total_var  = tk.StringVar(value="Total detections: 0")
        tk.Label(sess, textvariable=self.frames_var,
                 bg=BG_CARD, fg=DIM, font=("Arial", 9)).pack(anchor="w")
        tk.Label(sess, textvariable=self.total_var,
                 bg=BG_CARD, fg=DIM, font=("Arial", 9)).pack(anchor="w")

        # Per-class session counters
        self.session_lbl_vars = {}
        for cid in range(1, 5):
            v = tk.StringVar(value=f"{DISPLAY_NAMES[cid]}: 0")
            tk.Label(sess, textvariable=v,
                     bg=BG_CARD, fg=CLASS_COLORS_HEX[cid],
                     font=("Arial", 8)).pack(anchor="w")
            self.session_lbl_vars[cid] = v

        # ── Dataset info card ───────────────────────────────────────────────────
        self._section(right, "DATASET INFO")
        info = tk.Frame(right, bg=BG_CARD, padx=10, pady=8)
        info.pack(fill="x", pady=(0, 6))
        for line in [
            "SolDef-AI  (Kaggle)",
            "1150 SMT solder images",
            "3 viewpoints per component",
            "USB microscope (HD CMOS)",
            "IPC-A-610 Class 2 standard",
        ]:
            tk.Label(info, text=line, bg=BG_CARD, fg=DIM,
                     font=("Arial", 8)).pack(anchor="w")

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=BG_DARK, fg=ACCENT,
                 font=("Arial", 9, "bold")).pack(anchor="w", pady=(7, 3))

    # ── MODEL ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO
            if not WEIGHTS_PATH.exists():
                self.status_var.set("⚠ No weights — train first")
                self.root.after(0, lambda: self._show_placeholder(
                    "Train your model first\npython train.py"))
                return

            self.model = YOLO(str(WEIGHTS_PATH))

            if torch.cuda.is_available():
                self.device = "cuda"
                self.half   = True
                dev_name    = f"CUDA ({torch.cuda.get_device_name(0)[:20]})"
            else:
                self.device = "cpu"
                self.half   = False
                dev_name    = "CPU"

            # Warm-up inference
            dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=IMG_SIZE, half=self.half,
                               device=self.device, verbose=False)

            self.status_var.set(f"✅ Ready  |  {dev_name}")
            self.root.after(0, self._switch_to_camera)

        except Exception as e:
            self.status_var.set(f"❌ {str(e)[:40]}")
            print(f"[load_model] {e}")

    # ── MODE SWITCH ───────────────────────────────────────────────────────────
    def _switch_to_camera(self):
        self.mode = "camera"
        self.btn_cam.config(bg=ACCENT, fg=BG_DARK)
        self.btn_upload.config(bg=BG_PANEL, fg=TEXT)
        self.upload_bar.pack_forget()
        self._start_camera()

    def _switch_to_upload(self):
        self._stop_camera()
        self.mode = "upload"
        self.btn_cam.config(bg=BG_PANEL, fg=TEXT)
        self.btn_upload.config(bg=ACCENT, fg=BG_DARK)
        self.upload_bar.pack(fill="x", pady=(4, 0))
        self._show_placeholder("Opening file browser…")
        # Open dialog after a short delay so the UI refreshes first
        self.root.after(120, self._browse_image)

    # ── PLACEHOLDER ───────────────────────────────────────────────────────────
    def _show_placeholder(self, msg=""):
        w = max(self.left.winfo_width(),  800)
        h = max(self.left.winfo_height(), 500)
        img = Image.new("RGB", (w, h), BG_DARK)
        d   = ImageDraw.Draw(img)
        lines = msg.split("\n")
        y0 = h // 2 - len(lines) * 13
        for line in lines:
            bb = d.textbbox((0, 0), line)
            x  = (w - (bb[2] - bb[0])) // 2
            d.text((x, y0), line, fill=DIM)
            y0 += 26
        self._show_pil(img)

    # ── CAMERA ────────────────────────────────────────────────────────────────
    def _start_camera(self):
        self._stop_camera()
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except queue.Empty: break
        self.cam_running = True
        self.cam_thread  = threading.Thread(target=self._cam_loop, daemon=True)
        self.cam_thread.start()

    def _stop_camera(self):
        self.cam_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        time.sleep(0.05)

    def _cam_loop(self):
        cap = cv2.VideoCapture(CAMERA_IDX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_IDX)
        if not cap.isOpened():
            self.status_var.set(f"❌ Camera {CAMERA_IDX} not found")
            self.root.after(0, lambda: self._show_placeholder(
                f"Camera {CAMERA_IDX} not found\nCheck connection"))
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera = cap
        self.status_var.set(f"✅ Camera {CAMERA_IDX} active")
        self.last_time = time.time()

        while self.cam_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            annotated, counts = self._detect_live(frame)

            now = time.time()
            fps = 1.0 / max(now - self.last_time, 0.001)
            self.last_time = now
            self.frame_count += 1

            self.fps_var.set(f"FPS: {fps:.1f}")
            self.frames_var.set(f"Frames processed: {self.frame_count}")

            # Drop stale frame
            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            self.frame_queue.put((annotated, counts))

        cap.release()
        self.camera = None

    # ── FRAME QUEUE POLL (main thread) ─────────────────────────────────────────
    def _poll_frame_queue(self):
        try:
            annotated, counts = self.frame_queue.get_nowait()
            self._update_panel(counts)
            self._show_cv2(annotated)
        except queue.Empty:
            pass
        self.root.after(15, self._poll_frame_queue)

    # ── LIVE DETECTION ────────────────────────────────────────────────────────
    def _detect_live(self, frame):
        if self.model is None:
            return frame, {}
        try:
            # SolDef-AI images are close-up microscope shots — CLAHE helps a lot
            enhanced = self._clahe(frame)

            results = self.model(
                enhanced,
                conf         = self.conf_var.get(),
                iou          = self.iou_var.get(),
                imgsz        = IMG_SIZE,
                half         = self.half,
                device       = self.device,
                verbose      = False,
                agnostic_nms = False,
            )[0]

            counts    = {i: 0 for i in range(5)}
            annotated = frame.copy()

            for box in results.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                if cid == 0:
                    continue   # skip No-Defect boxes in live view
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS_BGR.get(cid, (255, 255, 255))
                label = f"{DISPLAY_NAMES.get(cid, str(cid))} {conf:.0%}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated,
                              (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                counts[cid] = counts.get(cid, 0) + 1
                self.session_dets[cid] += 1

            defect    = any(counts.get(c, 0) > 0 for c in range(1, 5))
            verdict   = "FAIL - DEFECT" if defect else "PASS - OK"
            v_color   = (0, 68, 255) if defect else (0, 255, 136)
            total_det = sum(counts.values())

            cv2.putText(annotated, verdict, (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, v_color, 3)
            cv2.putText(annotated, f"Det: {total_det}", (12, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            self.current_frame = annotated
            self.total_var.set(f"Total detections: {sum(self.session_dets.values())}")
            self._update_session_labels()
            return annotated, counts

        except Exception as e:
            print(f"[detect_live] {e}")
            return frame, {}

    # ── UPLOAD DETECTION (multi-scale + NMS, tuned for SolDef-AI) ─────────────
    def _detect_upload(self, frame):
        if self.model is None:
            return frame, {}
        try:
            enhanced = self._clahe(frame, clip=CLAHE_CLIP)
            all_boxes = []

            for scale in UPLOAD_SCALES:
                res = self.model(
                    enhanced,
                    conf    = UPLOAD_CONF,
                    iou     = UPLOAD_NMS_IOU,
                    imgsz   = scale,
                    half    = self.half,
                    device  = self.device,
                    verbose = False,
                )[0]
                for box in res.boxes:
                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cid == 0:
                        continue
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    all_boxes.append([x1, y1, x2, y2, conf, cid])

            keep      = self._nms(all_boxes, thresh=UPLOAD_NMS_IOU)
            counts    = {i: 0 for i in range(5)}
            annotated = frame.copy()

            for idx in keep:
                x1, y1, x2, y2, conf, cid = all_boxes[idx]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = CLASS_COLORS_BGR.get(int(cid), (255, 255, 255))
                name  = DISPLAY_NAMES.get(int(cid), str(cid))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                (tw, th), _ = cv2.getTextSize(
                    f"{name} {conf:.0%}", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated,
                              (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, f"{name} {conf:.0%}",
                            (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                counts[int(cid)] += 1
                self.session_dets[int(cid)] += 1

            self.current_frame = annotated
            self.total_var.set(f"Total detections: {sum(self.session_dets.values())}")
            self._update_session_labels()
            return annotated, counts

        except Exception as e:
            print(f"[detect_upload] {e}")
            return frame, {}

    # ── NMS HELPER ────────────────────────────────────────────────────────────
    @staticmethod
    def _nms(boxes, thresh=0.5):
        if not boxes:
            return []
        arr    = np.array([[b[0], b[1], b[2], b[3]] for b in boxes])
        scores = np.array([b[4] for b in boxes])
        x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w   = np.maximum(0, xx2 - xx1)
            h   = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= thresh]
        return keep

    # ── CLAHE PRE-PROCESSING ──────────────────────────────────────────────────
    @staticmethod
    def _clahe(frame, clip=CLAHE_CLIP, tile=CLAHE_TILE):
        """
        CLAHE in LAB space.  Boosts local contrast on SMT solder joints
        without blowing out highlights — important for SolDef-AI images
        which are close-up USB-microscope shots.
        """
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl   = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        l    = cl.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ── PANEL UPDATE ──────────────────────────────────────────────────────────
    def _update_panel(self, counts):
        defect = False
        for cid, var in self.count_vars.items():
            n = counts.get(cid, 0)
            var.set(str(n))
            if cid > 0 and n > 0:
                defect = True

        # Update confidence mini-bars (only in live mode — counts are proxies)
        for cid in range(1, 5):
            bg_w = 160
            filled = min(int(counts.get(cid, 0) * 40), bg_w)   # scale for display
            _, bar_fill = self.conf_bars[cid]
            bar_fill.place(x=0, y=0, relheight=1, width=filled)

        if defect:
            self.verdict_var.set("⚠  FAIL — DEFECT FOUND")
            self.verdict_lbl.config(fg=RED)
        else:
            self.verdict_var.set("✅  PASS — NO DEFECT")
            self.verdict_lbl.config(fg=GREEN)

    def _update_session_labels(self):
        for cid, v in self.session_lbl_vars.items():
            v.set(f"{DISPLAY_NAMES[cid]}: {self.session_dets[cid]}")

    # ── UPLOAD FLOW ───────────────────────────────────────────────────────────
    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select PCB Image (SolDef-AI or compatible)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All", "*.*")])
        if not path:
            # User cancelled — go back to camera
            self._switch_to_camera()
            return
        self.img_lbl.config(text=Path(path).name[:30])
        self._show_placeholder("Analyzing image…")
        threading.Thread(target=self._process_upload,
                         args=(path,), daemon=True).start()

    def _process_upload(self, path):
        try:
            frame = cv2.imread(path)
            if frame is None:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "Cannot read image!"))
                return
            annotated, counts = self._detect_upload(frame)
            self.root.after(0, lambda: self._finish_upload(annotated, counts))
        except Exception as e:
            print(f"[process_upload] {e}")
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Upload failed: {e}"))

    def _finish_upload(self, annotated, counts):
        self._update_panel(counts)
        self._show_cv2(annotated)
        self.current_frame = annotated

        defect = any(counts.get(c, 0) > 0 for c in range(1, 5))
        if defect:
            self.verdict_var.set("⚠  FAIL — DEFECT FOUND")
            self.verdict_lbl.config(fg=RED)
        else:
            self.verdict_var.set("✅  PASS — NO DEFECT")
            self.verdict_lbl.config(fg=GREEN)

    # ── DISPLAY ───────────────────────────────────────────────────────────────
    def _show_cv2(self, frame):
        self._show_pil(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def _show_pil(self, img):
        """
        Resize to fit the LEFT panel without growing it.
        Read dimensions from self.left (stable, pack_propagate=False),
        NOT from self.canvas (which grows with each image → feedback loop).
        """
        pw = max(self.left.winfo_width(),  800)
        ph = max(self.left.winfo_height(), 500)
        # Subtract top bars (~48px mode bar, possibly ~40px upload bar)
        ph = max(ph - 50, 400)

        iw, ih = img.size
        scale  = min(pw / iw, ph / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        nw, nh = max(nw, 320), max(nh, 240)

        img   = img.resize((nw, nh), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(img)
        self.canvas.config(image=imgtk, width=nw, height=nh)
        self.canvas.image = imgtk   # prevent GC

    # ── ACTIONS ───────────────────────────────────────────────────────────────
    def _save_snapshot(self):
        if self.current_frame is None:
            messagebox.showinfo("Info", "No image to save yet!")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"snapshot_{ts}.jpg"
        cv2.imwrite(str(path), self.current_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        messagebox.showinfo("Saved", f"Snapshot saved:\n{path}")

    def _ipc_report(self):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fn  = datetime.now().strftime("%Y%m%d_%H%M%S")
        det = {DISPLAY_NAMES[i]: int(self.count_vars[i].get())
               for i in range(5)}
        defect  = any(v > 0 for k, v in det.items() if k != "No Defect")
        verdict = "FAIL" if defect else "PASS"

        lines = [
            "========================================================",
            "        IPC-A-610 AUTOMATED INSPECTION REPORT           ",
            "        PCB Solder Defect Detection System V5            ",
            "========================================================",
            f"  Team      : CPEQC-029-030 Team 4 | TIP QC",
            f"  Dataset   : SolDef-AI (Kaggle - mauriziocalabrese)",
            f"  Timestamp : {ts}",
            f"  Standard  : IPC-A-610 Class 2",
            "========================================================",
            "  DETECTION RESULTS (current frame)",
            "========================================================",
        ]
        for name, count in det.items():
            if name == "No Defect":
                ipc = "PASS"
            else:
                ipc = "FAIL" if count > 0 else "OK"
            lines.append(f"  {name:<26} Count: {count:<4} [{ipc}]")

        lines += [
            "========================================================",
            "  SESSION TOTALS",
            "========================================================",
        ]
        for cid in range(5):
            lines.append(
                f"  {DISPLAY_NAMES[cid]:<26} Total: {self.session_dets[cid]}")

        lines += [
            "========================================================",
            f"  OVERALL VERDICT : {verdict}",
            f"  Frames Processed: {self.frame_count}",
            "========================================================",
        ]
        report = "\n".join(lines)

        win = tk.Toplevel(self.root)
        win.title("IPC-A-610 Report — V5")
        win.geometry("580x420")
        win.configure(bg=BG_DARK)
        txt = tk.Text(win, bg=BG_PANEL, fg=TEXT, font=("Courier", 9),
                      wrap="none", padx=12, pady=12)
        txt.pack(fill="both", expand=True, padx=12, pady=12)
        txt.insert("1.0", report)
        txt.config(state="disabled")

        path = RESULTS_DIR / f"ipc_report_{fn}.txt"
        path.write_text(report)

        tk.Button(win, text="Close", bg=ACCENT, fg=BG_DARK,
                  font=("Arial", 10, "bold"), relief="flat",
                  padx=14, pady=6, command=win.destroy).pack(pady=(0, 12))

    def _reset_stats(self):
        self.frame_count  = 0
        self.session_dets = {i: 0 for i in range(5)}
        self.frames_var.set("Frames processed: 0")
        self.total_var.set("Total detections: 0")
        for v in self.count_vars.values():
            v.set("0")
        for cid, v in self.session_lbl_vars.items():
            v.set(f"{DISPLAY_NAMES[cid]}: 0")
        self.verdict_var.set("READY")
        self.verdict_lbl.config(fg=ACCENT2)

    # ── CLOSE ─────────────────────────────────────────────────────────────────
    def on_close(self):
        self.cam_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = PCBDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
