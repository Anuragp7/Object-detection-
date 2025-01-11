import torch
import os
import cv2
import random

class ObjectDetection:
    def __init__(self, model_name="yolov5s"):
        # Load the YOLOv5 model
        self.model = torch.hub.load("ultralytics/yolov5", model_name)
        self.model.conf = 0.25  # Confidence threshold

    def detect_objects(self, frame):
        # Perform object detection on the frame
        results = self.model(frame)
        return results

    def process_detection(self, results):
        detections = []
        # Convert results to pandas DataFrame
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            bbox = [row.get('xmin'), row.get('ymin'), row.get('xmax'), row.get('ymax')]
            confidence = row.get('confidence')
            label = row.get('name')

            if None not in bbox:  # Ensure bbox values exist
                detections.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "label": label,
                })
        return detections

    def generate_json(self, detections):
        output = []
        object_id_counter = 1
        subobject_id_counter = 1

        for detection in detections:
            # Generate a random sub-object for demonstration
            subobject_bbox = [
                random.uniform(detection["bbox"][0], detection["bbox"][2]),  # Random x1 within main bbox
                random.uniform(detection["bbox"][1], detection["bbox"][3]),  # Random y1 within main bbox
                random.uniform(detection["bbox"][0], detection["bbox"][2]),  # Random x2 within main bbox
                random.uniform(detection["bbox"][1], detection["bbox"][3])   # Random y2 within main bbox
            ]
            subobject_bbox = [max(0, val) for val in subobject_bbox]  # Ensure no negative values

            # Structure the main object and sub-object
            object_entry = {
                "object": detection["label"],
                "id": object_id_counter,
                "bbox": detection["bbox"],
                "subobject": {
                    "object": f"{detection['label']}_subobject",
                    "id": subobject_id_counter,
                    "bbox": subobject_bbox
                }
            }

            object_id_counter += 1
            subobject_id_counter += 1
            output.append(object_entry)

        return output
