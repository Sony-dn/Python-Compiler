from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "YOLO Object Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()