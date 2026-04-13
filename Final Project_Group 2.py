import cv2
import time
import pyttsx3
import threading
import winsound
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from datetime import datetime

class DriverSentinel:
    def __init__(self, root):
        self.root = root
        self.root.title("DROWSINESS DETECTOR")
        self.root.geometry("1150x850")
        self.root.configure(bg="#020617")

        # --- Detection Variables ---
        self.running = False
        self.cap = None
        self.base_x = self.base_y = None
        
        # State Counters
        self.eye_counter = 0
        self.focus_counter = 0  
        self.face_loss_counter = 0
        self.alarm_active = False 

        # --- Adaptive Thresholds ---
        self.mode = "HIGHWAY" 
        self.EYE_LIMIT = 12       
        self.FOCUS_LIMIT = 18     
        self.FACE_LOST_LIMIT = 25 
        
        # AI Logic
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def build_ui(self):
        # --- Header ---
        header = tk.Frame(self.root, bg="#0f172a", height=80)
        header.pack(fill="x")
        tk.Label(header, text="DROWSINESS DETECTOR", font=("Impact", 28), bg="#0f172a", fg="#38bdf8").pack(pady=10)

        main_frame = tk.Frame(self.root, bg="#020617")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # --- Video Panel ---
        self.video_container = tk.Frame(main_frame, bg="#1e293b", padx=5, pady=5)
        self.video_container.pack(side="left", fill="both", expand=True)
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # --- Control Panel ---
        right_panel = tk.Frame(main_frame, bg="#020617", width=300)
        right_panel.pack(side="right", fill="y", padx=(20, 0))

        self.btn_start = tk.Button(right_panel, text="START MONITORING", command=self.start_sys, 
                                   bg="#10b981", fg="white", font=("Segoe UI", 12, "bold"), relief="flat", pady=10)
        self.btn_start.pack(fill="x", pady=5)

        self.btn_stop = tk.Button(right_panel, text="STOP SYSTEM", command=self.stop_sys, 
                                  bg="#ef4444", fg="white", font=("Segoe UI", 10), relief="flat")
        self.btn_stop.pack(fill="x", pady=5)

        tk.Label(right_panel, text="SETTINGS", font=("Segoe UI", 10, "bold"), bg="#020617", fg="#94a3b8").pack(pady=(15, 5))
        
        self.btn_calib = tk.Button(right_panel, text="🎯 RESET CALIBRATION", command=self.reset_calib, 
                                   bg="#3b82f6", fg="white", font=("Segoe UI", 10), relief="flat")
        self.btn_calib.pack(fill="x", pady=2)

        self.btn_mode = tk.Button(right_panel, text="🛣 MODE: HIGHWAY", command=self.toggle_mode, 
                                   bg="#6366f1", fg="white", font=("Segoe UI", 10), relief="flat")
        self.btn_mode.pack(fill="x", pady=2)

        self.btn_clear = tk.Button(right_panel, text="🧹 CLEAR LOGS", command=lambda: self.log_area.delete('1.0', tk.END), 
                                   bg="#475569", fg="white", font=("Segoe UI", 10), relief="flat")
        self.btn_clear.pack(fill="x", pady=2)

        self.log_area = scrolledtext.ScrolledText(right_panel, bg="#0f172a", fg="#38bdf8", 
                                                 font=("Consolas", 9), height=15, bd=0)
        self.log_area.pack(pady=20, fill="both", expand=True)

        # --- Status Bar ---
        self.status_bar = tk.Label(self.root, text="SYSTEM READY", font=("Segoe UI", 18, "bold"), 
                                  bg="#1e293b", fg="white", pady=25)
        self.status_bar.pack(fill="x")

    def reset_calib(self):
        self.base_x = self.base_y = None
        self.log_event("Calibration Reset: Position recalibrated.")

    def toggle_mode(self):
        if self.mode == "HIGHWAY":
            self.mode = "CITY"
            self.EYE_LIMIT, self.FOCUS_LIMIT = 18, 25
            self.btn_mode.config(text="🏙 MODE: CITY", bg="#a855f7")
        else:
            self.mode = "HIGHWAY"
            self.EYE_LIMIT, self.FOCUS_LIMIT = 12, 18
            self.btn_mode.config(text="🛣 MODE: HIGHWAY", bg="#6366f1")
        self.log_event(f"Mode changed: {self.mode}")

    def log_event(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_area.see(tk.END)

    def trigger_alarm(self, message):
        if not self.alarm_active:
            self.alarm_active = True
            self.log_event(f"ALERT: {message}")
            
            def alarm_task():
                # Verbal Alert (Runs once)
                self.engine.say(message)
                self.engine.runAndWait()

                # Continuous Beeping Loop
                while self.alarm_active and self.running:
                    if self.mode == "HIGHWAY":
                        # EMERGENCY: Piercing, ultra-fast siren
                        winsound.Beep(3800, 150)
                        time.sleep(0.05)
                    else:
                        # CITY: Firm, double-pulse rhythm (Like a modern car alert)
                        # Pulse 1
                        winsound.Beep(1800, 200)
                        time.sleep(0.1)
                        # Pulse 2
                        winsound.Beep(1800, 200)
                        time.sleep(0.4) # Brief pause before repeating
                
            threading.Thread(target=alarm_task, daemon=True).start()

    def start_sys(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.log_event("MONITORING ONLINE.")
            self.process_loop()

    def stop_sys(self):
        self.running = False
        self.alarm_active = False
        if self.cap: self.cap.release()
        self.video_label.config(image="")
        self.status_bar.config(bg="#1e293b", text="SYSTEM STANDBY")

    def process_loop(self):
        if not self.running: return
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(160, 160))

        status_text, status_color, danger = "DRIVER FOCUSED", "#10b981", False
        alert_msg = ""

        if len(faces) > 0:
            self.face_loss_counter = 0
            for (x, y, w, h) in faces:
                cx, cy = x + w//2, y + h//2
                if self.base_x is None: self.base_x, self.base_y = cx, cy

                roi_eye = gray[y:y+int(h*0.6), x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_eye, 1.1, 12, minSize=(30, 30))
                self.eye_counter = self.eye_counter + 1 if len(eyes) < 2 else 0

                is_nodding = cy > (self.base_y + (h * 0.16))
                is_looking_away = abs(cx - self.base_x) > (w * 0.22)
                self.focus_counter = self.focus_counter + 1 if (is_nodding or is_looking_away) else 0

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
        else:
            self.face_loss_counter += 1
            self.eye_counter = self.focus_counter = 0

        # Decision Engine
        if self.eye_counter >= self.EYE_LIMIT:
            alert_msg, status_text, status_color, danger = "WAKE UP! EYES CLOSED.", " CRITICAL: DROWSINESS", "#ef4444", True
        elif self.focus_counter >= self.FOCUS_LIMIT:
            alert_msg, status_text, status_color, danger = "FOCUS ON THE ROAD!", " WARNING: LOSS OF FOCUS", "#f59e0b", True
        elif self.face_loss_counter >= self.FACE_LOST_LIMIT:
            alert_msg, status_text, status_color, danger = "DRIVER NOT VISIBLE!", " ERROR: NO FACE DETECTED", "#ef4444", True

        self.status_bar.config(text=status_text, bg=status_color)
        self.video_container.config(bg=status_color)
        
        if danger:
            self.trigger_alarm(alert_msg)
        else:
            if self.alarm_active and self.eye_counter == 0 and self.focus_counter == 0:
                self.alarm_active = False
                self.log_event("Safety Restored.")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.process_loop)

    def on_closing(self):
        self.stop_sys()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DriverSentinel(root)
    root.mainloop()