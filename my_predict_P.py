import cv2
from patched_yolo_infer import (
    visualize_results_usual_yolo_inference,
)

from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("/Users/hayhay/Documents/ultralytics-main/runs/detect/train11/weights/best.pt")

# load img
img_path = "/Users/hayhay/Documents/labeling/IMG_6267.jpg"
img = cv2.imread(img_path)

# for preview plt.imshow(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2BGRA));
# plt.show()
# Train the model on the COCO8 example dataset for 100 epochs
results = model(img_path)

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
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
