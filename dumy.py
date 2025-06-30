from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
import threading
from scipy.spatial import distance as dist
from imutils import face_utils
from fer import FER
import pyttsx3
import numpy as np
import winsound
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize global variables
camera_running = False
capture = None
ALARM_ON = False
COUNTER = 0
drowsiness_count = 0
phone_usage_count = 0

# Initialize detection tools
emotion_detector = FER()
phone_usage_notifier = pyttsx3.init()
phone_usage_notifier.setProperty('voice', phone_usage_notifier.getProperty('voices')[1].id)
phone_usage_notifier.setProperty('rate', 150)

# Load YOLO for phone detection
net = cv2.dnn.readNet(r"C:\Users\hrith\Desktop\AI PROJECT\yolov3.weights", r"C:\Users\hrith\Desktop\AI PROJECT\yolov3 (1).cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open(r"C:\Users\hrith\Desktop\AI PROJECT\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load dlib's facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\hrith\Desktop\AI PROJECT\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Alert functions
def play_alert_sound():
    while ALARM_ON:
        winsound.Beep(1000, 500)

def start_alert():
    global ALARM_ON
    if not ALARM_ON:
        ALARM_ON = True
        threading.Thread(target=play_alert_sound, daemon=True).start()

def stop_alert():
    global ALARM_ON
    ALARM_ON = False

# Detection functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(frame):
    global COUNTER, ALARM_ON, drowsiness_count
    EYE_AR_THRESH = 0.28
    EYE_AR_CONSEC_FRAMES = 10

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    start_alert()
                    drowsiness_count += 1
                    print("Drowsiness detected!")
        else:
            if ALARM_ON:
                stop_alert()
            COUNTER = 0

def detect_emotions(frame):
    emotions = emotion_detector.detect_emotions(frame)
    for emotion in emotions:
        bbox = emotion['box']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
        emotion_texts = [f"{e}: {score*100:.1f}%" for e, score in emotion['emotions'].items() if score > 0.1]
        y_offset = bbox[1] - 10
        for text in emotion_texts:
            cv2.putText(frame, text, (bbox[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset -= 15

def detect_phone_usage(frame):
    global phone_usage_count
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    cellphone_detected = False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 67:
                if not cellphone_detected:
                    cellphone_detected = True
                    phone_usage_notifier.say("Please don't use your phone while driving")
                    phone_usage_notifier.runAndWait()
                    phone_usage_count += 1
                    print("Phone usage detected!")

# Start camera
def start_camera():
    global capture, camera_running
    capture = cv2.VideoCapture(0)
    camera_running = True

    while camera_running:
        ret, frame = capture.read()
        if not ret:
            break

        detect_drowsiness(frame)
        detect_emotions(frame)
        detect_phone_usage(frame)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask route to start the camera stream
@app.route('/start_camera')
def start_camera_stream():
    return Response(start_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to stop the camera
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running, capture
    camera_running = False
    if capture:
        capture.release()
    stop_alert()
    return jsonify({"status": "Camera stopped"})

# Route to render HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

