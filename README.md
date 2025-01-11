# Object-detection-
# Real-Time Object Detection with YOLOv5

This project leverages the YOLOv5 object detection model to process video frames, detect objects in real-time, annotate the frames, and generate hierarchical JSON data containing detection results. It dynamically resizes the video frames to fit the user's screen and supports sub-object detection for demonstration purposes.

---

## Features

- *Real-Time Object Detection*: Detects and annotates objects in video frames using YOLOv5.
- *Dynamic Screen Scaling*: Automatically resizes video frames to fit the user's screen resolution.
- *Hierarchical JSON Output*: Generates structured JSON files with detected objects and their attributes, including sub-object data.
- *Interactive Display*: Displays annotated video frames with bounding boxes, labels, and confidence scores.
- *Configurable Confidence Threshold*: Allows customization of the detection threshold for better accuracy.

---

## Prerequisites

- *Python*: Version 3.6 or higher.
- *PyTorch*: Install according to your system's specifications ([PyTorch Installation Guide](https://pytorch.org/get-started/locally/)).
- *OpenCV*: Required for video processing and display.
- *YOLOv5*: Automatically loaded using torch.hub.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/yolov5-object-detection.git
   cd yolov5-object-detection
