import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from scipy.spatial import distance
from collections import deque
import threading
import argparse

# Constants

MODEL_PATH = "C:/Users/Hp/Desktop/frt/mobilenet_v3_small.tflite"  # Update if different path
FACE_DIR = "faces"
FRAME_DIR = "frames"
FIDUCIAL_THRESHOLD = 0.05
MIN_DETECTION_CONFIDENCE = 0.8
MAX_NUM_FACES = 1
IMAGE_WIDTH = 640  # Define target width and height for image scaling
IMAGE_HEIGHT = 480
MIN_SHARPNESS = 70.0


class FaceRecognition:
    def __init__(self, camera_source, model_path=MODEL_PATH, face_dir=FACE_DIR, frame_dir=FRAME_DIR, camera_id=0):
        """
        Initializes the FaceRecognition object.

        Args:
            camera_source (str or int): The camera source, can be an RTSP URL (str) or a camera index (int).
            model_path (str): Path to the TFLite model.
            face_dir (str): Directory to save face crops.
            frame_dir (str): Directory to save best frames.
            camera_id (int):  An ID to identify the camera source.
        """
        self.camera_source = camera_source
        self.model_path = model_path
        self.face_dir = face_dir
        self.frame_dir = frame_dir
        self.camera_id = camera_id
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        self.face_data = {}
        self.next_person_id = 0
        self.recent_embeddings = {}
        self.face_positions = {}
        self.frame_count = 0

        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        # Load MobileNetV3 Model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Create Directories
        os.makedirs(self.face_dir, exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)

        # Initialize Face Detection and Face Mesh models
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1,
                                                                    min_detection_confidence=MIN_DETECTION_CONFIDENCE)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=MAX_NUM_FACES,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=MIN_DETECTION_CONFIDENCE)

        # Initialize Video Capture
        try:
            self.cap = cv2.VideoCapture(self.camera_source)  # Use camera_source
            if not self.cap.isOpened():
                raise IOError(f"Cannot open camera source: {self.camera_source}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        except Exception as e:
            raise IOError(f"Error opening camera source {self.camera_source}: {e}")

    def calculate_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def get_fiducial_score(self, image, face_landmarks):
        h, w, _ = image.shape
        key_points = {
            "left_eye": (face_landmarks[33].x * w, face_landmarks[33].y * h),
            "right_eye": (face_landmarks[263].x * w, face_landmarks[263].y * h),
            "nose_tip": (face_landmarks[1].x * w, face_landmarks[1].y * h),
            "left_mouth": (face_landmarks[61].x * w, face_landmarks[61].y * h),
            "right_mouth": (face_landmarks[291].x * w, face_landmarks[291].y * h),
        }
        eye_distance = distance.euclidean(key_points["left_eye"], key_points["right_eye"])
        nose_mid_x = (key_points["left_eye"][0] + key_points["right_eye"][0]) / 2
        return abs(nose_mid_x - key_points["nose_tip"][0]) / eye_distance

    def camera_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error reading frame from camera source: {self.camera_source}")
                break
            with self.lock:
                self.frame = frame.copy()
                self.frame_count += 1

        self.cap.release()

    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)

        # Create a copy for drawing that won't affect the saved images
        display_image = image.copy()

        detected_faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
                detected_faces.append((x, y, w, h))

        assigned_ids = []
        for x, y, w, h in detected_faces:
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                continue

            PADDING_PERCENTAGE = 0.2  # Increase or decrease as needed

            # Calculate padding
            x_pad = int(w * PADDING_PERCENTAGE)
            y_pad = int(h * PADDING_PERCENTAGE)

            # Expand the bounding box with padding
            x_start = max(0, x - x_pad)
            y_start = max(0, y - y_pad)
            x_end = min(image.shape[1], x + w + x_pad)
            y_end = min(image.shape[0], y + h + y_pad)

            # Crop the expanded face region
            face_crop = image[y_start:y_end, x_start:x_end]
            if face_crop.size == 0:
                continue

            # NEW CHECKS FOR roi size!
            if (x >= 0 and y >= 0 and (x + w) <= image_rgb.shape[1] and (y + h) <= image_rgb.shape[0] and w > 0 and h > 0):
                face_roi = image_rgb[y:y + h, x:x + w]
            else:
                continue

            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:  # Check if ROI is empty
                continue

            face_mesh_results = self.face_mesh.process(face_roi)
            if not face_mesh_results.multi_face_landmarks:
                continue

            fiducial_score = self.get_fiducial_score(image, face_mesh_results.multi_face_landmarks[0].landmark)
            if fiducial_score > FIDUCIAL_THRESHOLD:
                continue

            face_resized = cv2.resize(face_crop, (224, 224))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], face_input)
            self.interpreter.invoke()
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            assigned_id = None
            min_distance = float("inf")
            for person_id, data in self.face_data.items():
                similarity = self.cosine_similarity(embedding, data['embedding'])
                spatial_distance = np.linalg.norm(np.array([x, y]) - np.array(self.face_positions.get(person_id, (0, 0))))
                if similarity > 0.7 and spatial_distance < 100:
                    if spatial_distance < min_distance:
                        min_distance = spatial_distance
                        assigned_id = person_id

            if assigned_id is None:
                assigned_id = self.next_person_id
                self.face_data[assigned_id] = {'embedding': embedding, 'best_face': None, 'sharpness': 0,
                                                 'best_frame': None}
                self.recent_embeddings[assigned_id] = deque(maxlen=5)
                self.next_person_id += 1

            self.recent_embeddings[assigned_id].append(embedding)
            averaged_embedding = np.mean(self.recent_embeddings[assigned_id], axis=0)
            self.face_data[assigned_id]['embedding'] = averaged_embedding
            self.face_positions[assigned_id] = (x, y)

            sharpness = self.calculate_sharpness(face_crop)
            if sharpness > self.face_data[assigned_id]['sharpness']:
                if sharpness > MIN_SHARPNESS:
                    self.face_data[assigned_id]['sharpness'] = sharpness
                    self.face_data[assigned_id]['best_face'] = face_crop
                    self.face_data[assigned_id]['best_frame'] = image.copy()  # Store the original frame without bounding boxes

            assigned_ids.append(assigned_id)
            # Draw rectangles and text only on the display image, not the original
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_image, f"ID: {assigned_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        return display_image  # Return the display image for showing on screen

    def processing_thread(self):
        while self.running:
            with self.lock:
                if self.frame is None:
                    continue
                current_frame = self.frame.copy()

            processed_frame = self.process_frame(current_frame)

            cv2.imshow('Face Detection & ID Assignment', cv2.flip(processed_frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                self.running = False
                break

    def save_faces(self):
        for person_id, data in self.face_data.items():
            if data['best_face'] is not None and data['best_frame'] is not None:
                face_filename = os.path.join(self.face_dir, f"person_{person_id}_{self.camera_id}.jpg")
                frame_filename = os.path.join(self.frame_dir, f"person_{person_id}_{self.camera_id}.jpg")

                cv2.imwrite(face_filename, data['best_face'])
                cv2.imwrite(frame_filename, data['best_frame'])

                print(f"Saved best face and frame for Person {person_id} from camera {self.camera_id} to: {face_filename} and {frame_filename}")


    def run(self):
        thread1 = threading.Thread(target=self.camera_thread)  # Changed ffmpeg_camera_thread to camera_thread
        thread2 = threading.Thread(target=self.processing_thread)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        self.save_faces()
        cv2.destroyAllWindows()
        self.face_detection.close()
        self.face_mesh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition using Webcam or RTSP Stream")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (default 0) or RTSP URL (e.g., rtsp://username:password@ip:port/path)")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID for saving files.")
    args = parser.parse_args()

    try:
        camera_source = int(args.source)  # Try converting to integer (camera index)
    except ValueError:
        camera_source = args.source  # If not an integer, assume it's an RTSP URL


    face_recognition = FaceRecognition(camera_source=camera_source, camera_id=args.camera_id)
    face_recognition.run()