import cv2

from ultralytics import YOLO

model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")

results = model.predict(source=0, show=True)

print(results)

if cv2.waitKey(1) == ord("q"):
    cv2.destroyAllWindows()
