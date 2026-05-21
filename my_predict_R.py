import cv2

from ultralytics import YOLO


def prediction_R(root):
    model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("YOLO Real-Time", annotated)
        if cv2.waitKey(1) & 0xFF == ord("z"):
            root.deiconify()
            break
    cap.release()
    cv2.destroyAllWindows()
