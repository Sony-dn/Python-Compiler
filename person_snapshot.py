from ultralytics import YOLO
import cv2, os
from datetime import datetime

model, cap = YOLO("yolov8n.pt"), cv2.VideoCapture(0)
os.makedirs("snapshots", exist_ok=True)
last_save = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    r = model(frame)[0]
    person_count = sum(int(b.cls[0]) == 0 for b in r.boxes)
    # names = model.names
    # person_count = sum(names[int(b.cls[0])] == "person" for b in r.boxes)
    # car_or_phone = any(names[int(b.cls[0])] in ["car", "cell phone"] for b in r.boxes)

    frame = r.plot(0)
    cv2.putText(frame, f"Persons: {person_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if person_count and person_count<2 and (datetime.now().timestamp() - last_save) >= 3:
    # if person_count and car_or_phone and (datetime.now().timestamp() - last_save) >= 3:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"{timestamp} | Persons: {person_count}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
        print(f"✅ Saved snapshot")
        last_save = datetime.now().timestamp()

    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()






