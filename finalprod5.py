import torch
import cv2
import numpy as np
import time
from yolov5.utils.general import check_img_size, non_max_suppression, set_logging, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from collections import defaultdict
import os

# Define the plot_one_box function manually
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [255, 0, 0]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

set_logging()
device = select_device('')  # Select device (0 for CPU or '0' or '0,1,2,3' for GPU)

# Load model
model = torch.load('yolov5_updated_5.pt', map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

# Specify the classes you want to detect
target_classes = ['plastic bottle', 'wine glass', 'garbage container', 'fork', 'knife', 'book', 'spoon', 'banana', 'apple', 'orange', 'garbage_container']
class_indices = [model.names.index(cls) for cls in target_classes]

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam")
    exit()

# List to store the points of the polygon
points = []
roi_contour = None
tracking_info = defaultdict(lambda: {'start_time': None, 'last_seen': None, 'snapshot_taken': False, 'next_snapshot_time': None})

# Mouse callback function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global points, roi_contour
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point on left mouse button click
        points.append((x, y))
        print(f"Point added: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            roi_contour = np.array(points)
            points = []

# Create a window and set the mouse callback
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', draw_polygon)

roi_defined = False  # Flag to check if ROI has been defined

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLOv5 object detection
    img = letterbox(frame, 640, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=class_indices, agnostic=False)

    # Process detections
    current_time = time.time()
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=[255, 0, 0], line_thickness=3)

                # Tracking logic
                obj_id = f"{int(xyxy[0])}_{int(xyxy[1])}_{int(cls)}"
                if obj_id not in tracking_info:
                    tracking_info[obj_id]['start_time'] = current_time
                tracking_info[obj_id]['last_seen'] = current_time

                # Check proximity to ROI
                if roi_contour is not None and roi_defined:
                    center_x = int((xyxy[0] + xyxy[2]) / 2)
                    center_y = int((xyxy[1] + xyxy[3]) / 2)
                    if cv2.pointPolygonTest(roi_contour, (center_x, center_y), True) >= 0:
                        tracking_info[obj_id]['start_time'] = current_time
                        tracking_info[obj_id]['snapshot_taken'] = False
                        tracking_info[obj_id]['next_snapshot_time'] = None

    # Check for objects outside ROI for more than 15 seconds and handle snapshot logic
    if roi_defined:
        for obj_id, info in tracking_info.items():
            if current_time - info['start_time'] > 15 and not info['snapshot_taken']:
                # Save snapshot
                snapshot_path = f"snapshot_{obj_id}_{int(current_time)}.png"
                cv2.imwrite(snapshot_path, frame)
                print(f"Snapshot taken for {obj_id} and saved to {snapshot_path}")
                tracking_info[obj_id]['snapshot_taken'] = True
                tracking_info[obj_id]['next_snapshot_time'] = current_time + 600  # 10 minutes later
            elif info['snapshot_taken'] and info['next_snapshot_time'] and current_time >= info['next_snapshot_time']:
                # Save snapshot again after 10 minutes
                snapshot_path = f"snapshot_{obj_id}_{int(current_time)}.png"
                cv2.imwrite(snapshot_path, frame)
                print(f"Another snapshot taken for {obj_id} and saved to {snapshot_path}")
                tracking_info[obj_id]['next_snapshot_time'] = current_time + 600  # 10 minutes later

    # Draw the temporary polygon while drawing
    temp_frame = frame.copy()
    if points:
        if len(points) > 1:
            cv2.polylines(temp_frame, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        for point in points:
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
    elif roi_contour is not None:
        cv2.polylines(temp_frame, [roi_contour], isClosed=True, color=(0, 255, 0), thickness=2)

    end_time = time.time()
    inference_time = end_time - start_time
    cv2.putText(temp_frame, f"Inference time: {inference_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', temp_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break
    elif key == ord('w'):  # Press 'w' to complete the polygon and enable snapshot logic
        if len(points) > 1:
            roi_contour = np.array(points)
            points = []
            roi_defined = True  # Enable snapshot logic

cap.release()
cv2.destroyAllWindows()
