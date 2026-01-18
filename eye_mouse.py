import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque


pyautogui.FAILSAFE = False
SMOOTHING = 6
CLICK_COOLDOWN = 1.0


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


cam = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
last_click_time = 0


trail = deque(maxlen=15)


def gradient_ring(img, center, base_radius, colors, thickness=2):
    """Draws multiple circles to fake a gradient ring."""
    x, y = center
    for i, c in enumerate(colors):
        r = base_radius + i * 2
        cv2.circle(img, (x, y), r, c, thickness)

def neon_pulse(img, center, radius, color_main, color_glow, pulse):
    """Neon pulse glow effect."""
    x, y = center
    
    cv2.circle(img, (x, y), radius + 18 + pulse, color_glow, 2)
    cv2.circle(img, (x, y), radius + 12 + pulse, color_glow, 2)

   
    cv2.circle(img, (x, y), radius, color_main, -1)

    cv2.circle(img, (x - 2, y - 2), 2, (255, 255, 255), -1)

def draw_trail(img, trail_points):
    """Neon trail behind iris point."""
    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue
        thickness = int(np.interp(i, [1, len(trail_points)], [10, 2]))
        cv2.line(img, trail_points[i - 1], trail_points[i], (255, 0, 255), thickness)

print("✅ Ultra Eye Mouse Started... Press ESC to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    
    pulse = int(4 * abs(np.sin(time.time() * 3)))  

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        
        iris = landmarks[475]
        x = int(iris.x * w)
        y = int(iris.y * h)

        mouse_x = int((x / w) * screen_w)
        mouse_y = int((y / h) * screen_h)

       
        curr_x = prev_x + (mouse_x - prev_x) / SMOOTHING
        curr_y = prev_y + (mouse_y - prev_y) / SMOOTHING
        pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

      
        trail.appendleft((x, y))
        draw_trail(frame, trail)

        
        gradient_ring(
            frame,
            (x, y),
            base_radius=8,
            colors=[
                (255, 0, 255),     
                (255, 50, 200),    
                (255, 255, 0),     
                (0, 255, 255)      
            ],
            thickness=2
        )

        neon_pulse(
            frame,
            (x, y),
            radius=6,
            color_main=(0, 255, 255),     
            color_glow=(255, 0, 255),     
            pulse=pulse
        )

        
        top = landmarks[159]
        bottom = landmarks[145]

        top_pt = (int(top.x * w), int(top.y * h))
        bottom_pt = (int(bottom.x * w), int(bottom.y * h))

      
        neon_pulse(frame, top_pt, 4, (255, 0, 255), (255, 150, 255), pulse)
        neon_pulse(frame, bottom_pt, 4, (255, 0, 255), (255, 150, 255), pulse)

       
        eye_ratio = abs(top.y - bottom.y)

        if eye_ratio < 0.01:
            current_time = time.time()
            if current_time - last_click_time > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = current_time
                print("🖱️ Clicked")

   
    cv2.putText(frame, "Neon Eye Mouse (ESC to Exit)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("🔥 Ultra Neon Eye Mouse", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
