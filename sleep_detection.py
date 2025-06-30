# sleep_detection.py
import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import winsound
import time

# Global variable for storing drowsy and awake events
detection_results = []

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR) to detect drowsiness."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness():
    """Detect drowsiness and update `detection_results` with timestamps and durations."""
    EYE_AR_THRESH = 0.20
    EYE_AR_CONSEC_FRAMES = 15
    COUNTER = 0
    ALARM_ON = False
    drowsy_start_time = None  # To store the start time of each drowsy period

    predictor_path = "C:\\Users\\hrith\\Desktop\\AI PROJECT\\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

            ear = -1  # Default EAR if no faces are detected
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                print(f"EAR: {ear:.2f}")  # Debug EAR values

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    print(f"Drowsy counter: {COUNTER}")  # Debug counter

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            drowsy_start_time = time.time()  # Record start of drowsy period
                            winsound.Beep(1000, 500)
                            print("Drowsiness detected!")
                            detection_results.append({"status": "drowsy", "time": drowsy_start_time})
                else:
                    if ALARM_ON:
                        drowsy_end_time = time.time()  # Record end of drowsy period
                        drowsy_duration = drowsy_end_time - drowsy_start_time
                        detection_results.append({"status": "awake", "time": drowsy_end_time, "duration": drowsy_duration})
                        print(f"Drowsiness period lasted for {drowsy_duration:.2f} seconds.")
                    COUNTER = 0
                    ALARM_ON = False

            # Display EAR on the frame
            if ear != -1:
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.imshow("Sleep Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
