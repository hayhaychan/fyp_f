import builtins
import csv
import tkinter as tk
from datetime import datetime

import cv2
from patched_yolo_infer import (
    visualize_results_usual_yolo_inference,
)

from ultralytics import YOLO

label1 = "Welcome to my final year project\n I'm Chan Lok Hei 22068341D\n This project aims to help enhancing the speed of classifying garbage."


def open():
    print("open")


def exit():
    print("exit")
    root.destroy()


def recreate():
    global root
    root.update()
    root.deiconify()


def prediction_R():
    global root
    root.withdraw()
    model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train14/weights/best.pt")
    cap = cv2.VideoCapture(0)
    session_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Get the ID (number) and convert it to a Name (string)
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf_score = round(float(box.conf[0]) * 100, 2)
                session_data.append([class_name, conf_score])

        annotated = results[0].plot()
        cv2.imshow("YOLO Real-Time", annotated)
        if cv2.waitKey(1) & 0xFF == ord("z"):
            cv2.destroyAllWindows()
            save_accuracy_data(session_data)
            break
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()


def prediction_P():
    global root
    root.withdraw()
    model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")
    img_path = "/Users/hayhay/Documents/labeling/IMG_6267.jpg"
    img = cv2.imread(img_path)
    results = model(img_path)
    for result in results:
        if cv2.waitKey(1) == ord("z"):
            cv2.destroyAllWindows()
            break
        print(result)
        imgsz = 640
        conf = 0.4
        iou = 0.7

        visualize_results_usual_yolo_inference(
            img,
            model,
            imgsz,
            conf,
            iou,
            segment=True,
            show_boxes=True,
            show_class=True,
            fill_mask=False,
            alpha=0.3,
            color_class_background=(0, 0, 255),
            color_class_text=(255, 255, 255),
            thickness=4,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1.5,
            delta_colors=3,
            dpi=150,
            random_object_colors=False,
            show_confidences=False,
            axis_off=True,
            show_classes_list=[],
            list_of_class_colors=None,
            return_image_array=False,
            inference_extra_args=None,
        )
        cv2.destroyAllWindows()

    root.deiconify()


def save_accuracy_data(data):
    filename = f"accuracylog{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    # Use io.open instead of just open
    with builtins.open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Object_Class", "Accuracy_Score"])
        writer.writerows(data)
    print(f"Data saved to {filename}")


# create main
root = tk.Tk()
root.title("Project Garbage-classification")
root.geometry("700x600")

menu = tk.Menu(root)
menubar = tk.Menu(menu)
menubar.add_command(label="Open", command=root.deiconify)
menubar.add_command(label="Exit", command=exit)


# adding elements
label = tk.Label(root, text=label1, font=("Arial", 16))
label.pack(pady=150)
button1 = tk.Button(root, text="Real-Time Mode", command=prediction_R)
button1.pack(side="left", padx=60)
button2 = tk.Button(root, text="Photo Analysis Mode", command=prediction_P)
button2.pack(side="right", padx=60)
buttonQuit = tk.Button(root, text="quit", command=quit)
buttonQuit.pack(side="bottom", padx=20)

# main
root.mainloop()
