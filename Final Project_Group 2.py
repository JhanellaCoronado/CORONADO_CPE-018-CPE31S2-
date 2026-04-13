import cv2
import winsound
import time
from datetime import datetime

# --- SETTINGS ---
LOG_FILE = "driver_safety_log.txt"
EYE_LIMIT = 15          # Frames for closed eyes
TILT_LIMIT = 20         # Frames for head tilt/nodding
FACE_LOSS_LIMIT = 40    # Frames before alert if driver disappears
ALARM_FREQ = 2500       
ALARM_DURATION = 300 

# --- INITIALIZATION ---
eye_counter = 0
tilt_counter = 0
face_loss_counter = 0
alarm_active = False
base_y = None 

def log_event(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] ALERT: {msg}\n")

# Load Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

print("System Starting... Monitoring Eyes and Head Position.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(160, 160))
    
    eyes_found = False
    head_tilted = False

    if len(faces) > 0:
        face_loss_counter = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
            
            # --- CALIBRATION ---
            if base_y is None:
                base_y = y
            
            # --- 1. HEAD TILT (NODDING) ---
            if y > (base_y + (h * 0.20)): 
                head_tilted = True
                cv2.putText(frame, "DROWSY: HEAD NOD", (x, y-10), 1, 1.5, (0, 0, 255), 2)

            # --- 2. EYE DETECTION (CLOSED EYES) ---
            roi_gray_eyes = gray[y:y+int(h*0.6), x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray_eyes, 1.1, 10, minSize=(30, 30))
            if len(eyes) >= 2: 
                eyes_found = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
    else:
        face_loss_counter += 1

    # --- LOGIC FILTER ---
    if len(faces) > 0:
        eye_counter = eye_counter + 1 if not eyes_found else 0
        tilt_counter = tilt_counter + 1 if head_tilted else 0
    else:
        eye_counter = 0
        tilt_counter = 0

    # --- ALERT TRIGGER ---
    drowsy_eyes = eye_counter >= EYE_LIMIT
    drowsy_tilt = tilt_counter >= TILT_LIMIT
    critical_face_loss = face_loss_counter >= FACE_LOSS_LIMIT

    if drowsy_eyes or drowsy_tilt or critical_face_loss:
        cv2.rectangle(frame, (0, 0), (w_frame, 60), (0, 0, 255), -1)
        
        if critical_face_loss: msg = "DRIVER NOT VISIBLE!"
        elif drowsy_eyes: msg = "WAKE UP! EYES CLOSED!"
        else: msg = "HEAD DROPPED! STAY ALERT!"
            
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        winsound.Beep(ALARM_FREQ, ALARM_DURATION)
        
        if not alarm_active:
            log_event(msg)
            alarm_active = True
    else:
        cv2.rectangle(frame, (0, 0), (w_frame, 60), (0, 255, 0), -1)
        cv2.putText(frame, "STATUS: MONITORING ACTIVE", (120, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        alarm_active = False

    # --- DASHBOARD ---
    cv2.putText(frame, f"Eye Closure: {eye_counter}", (10, h_frame - 40), 1, 1, (255, 255, 255), 1)
    cv2.putText(frame, f"Head Tilt: {tilt_counter}", (10, h_frame - 15), 1, 1, (255, 255, 255), 1)

    cv2.imshow("RoadSafety AI - Eye & Tilt Only", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()