import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import winsound
from fer import FER
import pyttsx3
import numpy as np
import time
import threading

# Initialize FER for emotion detection
emotion_detector = FER()

# Initialize text-to-speech for phone usage alert
phone_usage_notifier = pyttsx3.init()
phone_usage_notifier.setProperty('voice', phone_usage_notifier.getProperty('voices')[1].id)
phone_usage_notifier.setProperty('rate', 150)

# Load YOLO model and classes for phone detection
net = cv2.dnn.readNet("C:\\Users\\hrith\\Desktop\\AI PROJECT\\yolov3.weights", "C:\\Users\\hrith\\Desktop\\AI PROJECT\\yolov3 (1).cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("C:\\Users\\hrith\\Desktop\\AI PROJECT\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load dlib's facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\hrith\\Desktop\\AI PROJECT\\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Global counters
drowsiness_count = 0
phone_usage_count = 0
ALARM_ON = False
COUNTER = 0

def play_alert_sound():
    """Play a continuous alert sound in a separate thread."""
    while ALARM_ON:
        winsound.Beep(1000, 500)  # Sound frequency and duration

def start_alert():
    """Start the alert sound if not already playing."""
    global ALARM_ON
    if not ALARM_ON:
        ALARM_ON = True
        threading.Thread(target=play_alert_sound, daemon=True).start()

def stop_alert():
    """Stop the alert sound."""
    global ALARM_ON
    ALARM_ON = False

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR) to detect drowsiness."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(frame):
    """Detects drowsiness based on EAR."""
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

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

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

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def detect_emotions(frame):
    """Detect emotions in a given frame."""
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
    """Detects phone usage using YOLO object detection."""
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

            if confidence > 0.5 and class_id == 67:  # Class 67 is for cell phones
                if not cellphone_detected:
                    cellphone_detected = True
                    phone_usage_notifier.say("Please don't use your phone while driving")
                    phone_usage_notifier.runAndWait()
                    phone_usage_count += 1
                    print("Phone usage detected!")

                # Draw bounding box around phone
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Phone Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def main():
    """Main function to run all detections concurrently."""
    global ALARM_ON
    ALARM_ON = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Run all three detections on the same frame
            detect_drowsiness(frame)
            detect_emotions(frame)
            detect_phone_usage(frame)

            # Display the processed frame
            cv2.imshow("Integrated Detection System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_alert()  # Ensure the alert stops if the program is exited
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
