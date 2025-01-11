import cv2
import json
import ctypes
from object_detection import ObjectDetection

def process_video(video_path, detector):
    # Get screen resolution dynamically
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)  # Screen width
    screen_height = user32.GetSystemMetrics(1)  # Screen height
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    output_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in the frame
        results = detector.detect_objects(frame)
        
        # Process detection results
        detections = detector.process_detection(results)
        
        # Generate the hierarchical JSON data
        frame_json = detector.generate_json(detections)
        output_data.extend(frame_json)
        
        # Annotate the frame
        for detection in detections:
            bbox = detection["bbox"]
            label = detection["label"]
            confidence = detection["confidence"]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Resize the frame to fit the screen dynamically
        frame_resized = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
        
        # Display the resized frame
        cv2.imshow("Video", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write detection results to a JSON file
    with open("output.json", "w") as f:
        json.dump(output_data, f, indent=4)
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "test_video.mp4"  # Replace with your video file path
    detector = ObjectDetection()
    process_video(video_path, detector)

if __name__ == "__main__":
    main()
