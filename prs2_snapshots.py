
# from ultralytics import YOLO
# import cv2, os
# from datetime import datetime

# model, cap = YOLO("yolov8n.pt"), cv2.VideoCapture(0)
# os.makedirs("snapshots", exist_ok=True)
# last_save = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     r = model(frame)[0]
#     frame = r.plot()

#     # Detect objects
#     person_detected = False
#     car_detected = False
#     phone_detected = False

#     for b in r.boxes:
#         cls_id = int(b.cls[0])

#         if cls_id == 0:   # person
#             person_detected = True
#         elif cls_id == 2: # car
#             car_detected = True
#         elif cls_id == 67: # cellphone
#             phone_detected = True

#     # Condition: person + (car or phone)
#     match_detected = person_detected and (car_detected or phone_detected)

#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # Display info bar
#     cv2.rectangle(frame, (0, 0), (500, 60), (0, 0, 0), -1)

#     status = "No match"
#     if match_detected:
#         status = "Person with Car/Phone!"

#     cv2.putText(frame, f"{timestamp} | {status}",
#                 (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Save snapshot only when condition is true
#     if match_detected and (datetime.now().timestamp() - last_save) >= 3:
#         filename = f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#         cv2.imwrite(filename, frame)
#         print("✅ Saved snapshot:", filename)
#         last_save = datetime.now().timestamp()

#     cv2.imshow("Person + Car/Phone Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2, os
from datetime import datetime

# Load YOLOv8 model and webcam
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Create folder to save snapshots
os.makedirs("snapshots", exist_ok=True)

car_was_detected = False  # Track previous frame state

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    r = model(frame)[0]
    frame = r.plot()

    # Detect car
    car_detected = False

    for b in r.boxes:
        cls_id = int(b.cls[0])
        if cls_id == 2:  # COCO class id for car
            car_detected = True
            break

    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Display info bar
    cv2.rectangle(frame, (0, 0), (500, 60), (0, 0, 0), -1)

    status = "No car detected"
    if car_detected:
        status = "Car Detected!"

    cv2.putText(
        frame,
        f"{timestamp} | {status}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # ✅ Save snapshot ONLY when car appears (False → True)
    if car_detected and not car_was_detected:
        filename = f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print("✅ Car appeared! Snapshot saved:", filename)

    # Update previous state
    car_was_detected = car_detected

    # Show output
    cv2.imshow("Car Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()