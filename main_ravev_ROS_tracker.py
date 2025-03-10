import cv2
import numpy as np
import os

class RAVEVTracker:
    def __init__(self, config_path, weights_path, labels_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            print("CUDA not available. Using CPU.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(labels_path, "r") as file:
            self.labels = file.read().strip().split("\n")

        self.emergency_vehicle_classes = ['ambulance', 'fire truck', 'truck']

    def emergency_vehicle_classifier(self, class_id):
        if self.labels[class_id] in self.emergency_vehicle_classes:
            return self.labels[class_id]
        return None

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return

        frame_count = 0
        frame_skip = 5

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                detections = self.detect_objects(frame)
                self.handle_tracks(detections, frame)

                resized_frame = cv2.resize(frame, (640, 360))
                cv2.imshow("RAVEV Tracker", resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Processing interrupted by user.")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def detect_objects(self, frame, confidence_threshold=0.3):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(layer_names)

        detections = []
        h, w = frame.shape[:2]
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    box = detection[:4] * np.array([w, h, w, h])
                    center_x, center_y, width, height = box.astype("int")
                    x_min = int(center_x - (width / 2))
                    y_min = int(center_y - (height / 2))
                    x_max = x_min + width
                    y_max = y_min + height

                    detections.append([x_min, y_min, x_max, y_max, confidence, class_id])

        return detections

    def handle_tracks(self, detections, frame):
        for detection in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = detection
            roi = frame[y_min:y_max, x_min:x_max]

            emergency_vehicle = self.emergency_vehicle_classifier(class_id)
            if emergency_vehicle:
                print(f"Emergency vehicle detected: Ambulance")
                color = (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                text = f"{emergency_vehicle}: {confidence:.2f}"
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                label = self.labels[class_id]
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == '__main__':
    config_path = r"C:\Users\Sairam\Documents\RAVEV-master[1]\RAVEV-master\yolov4.cfg"
    weights_path = r"C:\Users\Sairam\Documents\RAVEV-master[1]\RAVEV-master\yolov4.weights"
    labels_path = r"C:\Users\Sairam\Documents\RAVEV-master[1]\RAVEV-master\coco.names"

    tracker = RAVEVTracker(config_path, weights_path, labels_path)
    video_path = r"C:\Users\Sairam\Documents\RAVEV-master[1]\RAVEV-master\Ambulance struck in Chennai City Traffic..mp4"
    output_path = r"C:\Users\Sairam\Documents\RAVEV-master\Processed_Videos\ambulance_detected_output.mp4"

    tracker.process_video(video_path, output_path)
