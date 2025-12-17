import cv2
import numpy as np
import gradio as gr
import time

# -----------------------------
# Centroid Tracker
# -----------------------------
class CentroidTracker:
    def __init__(self):
        self.nextID = 0
        self.objects = {}

    def update(self, rects):
        centroids = []
        for (x, y, w, h) in rects:
            centroids.append((int(x + w / 2), int(y + h / 2)))

        new_objects = {}
        for centroid in centroids:
            minDist = float("inf")
            objectID = None

            for ID, oldCentroid in self.objects.items():
                dist = np.linalg.norm(np.array(centroid) - np.array(oldCentroid))
                if dist < minDist and dist < 40:
                    minDist = dist
                    objectID = ID

            if objectID is None:
                objectID = self.nextID
                self.nextID += 1

            new_objects[objectID] = centroid

        self.objects = new_objects
        return self.objects


# -----------------------------
# Load DNN Models
# -----------------------------
gender_net = cv2.dnn.readNetFromCaffe(
    "/home/teju_1209/jetson-inference/models/gendernet/gender_deploy.prototxt",
    "/home/teju_1209/jetson-inference/models/gendernet/gender_net.caffemodel"
)

age_net = cv2.dnn.readNetFromCaffe(
    "/home/teju_1209/jetson-inference/models/gendernet/age_deploy.prototxt",
    "/home/teju_1209/jetson-inference/models/gendernet/age_net.caffemodel"
)

GENDER_LIST = ["Male", "Female"]
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)","(21-24)"
            "(25-32)", "(38-43)", "(48-53)", "(60-8)", "(80-90)", "(90-100)"]

# -----------------------------
# Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    "/home/teju_1209/jetson-inference/haarcascade_frontalface_default.xml"
)

tracker = CentroidTracker()

# -----------------------------
# Streaming Function
# -----------------------------
def stream_camera():
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # camera warm-up

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        rects = [(x, y, w, h) for (x, y, w, h) in faces]
        objects = tracker.update(rects)

        # -----------------------------
        # People Count
        # -----------------------------
        people_count = len(objects)

        for objectID, centroid in objects.items():
            for (x, y, w, h) in faces:
                cx, cy = int(x + w / 2), int(y + h / 2)
                if abs(cx - centroid[0]) < 2 and abs(cy - centroid[1]) < 2:
                    face = frame[y:y + h, x:x + w]

                    blob = cv2.dnn.blobFromImage(
                        face, 1.0, (227, 227),
                        (78.426, 87.769, 114.896)
                    )

                    gender_net.setInput(blob)
                    gender = GENDER_LIST[gender_net.forward()[0].argmax()]

                    age_net.setInput(blob)
                    age = AGE_LIST[age_net.forward()[0].argmax()]

                    label = f"ID {objectID} | {gender} | Age {age}"

                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                    break

        # -----------------------------
        # Display People Count
        # -----------------------------
        cv2.putText(frame, f"People Count: {people_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

    cap.release()


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# ✈️ Airport Security System using Jetson Nano")
    gr.Markdown("### Live Face Detection, Age & Gender Classification with ID Tracking & People Counting")

    output = gr.Image(label="Live Airport Surveillance Feed")

    demo.load(stream_camera, outputs=output)

demo.launch()
