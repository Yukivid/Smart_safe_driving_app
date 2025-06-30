import cv2
from fer import FER
import time
from collections import deque

def initialize_emotion_detector():
    """Initialize the FER emotion detector and video capture."""
    emotion_detector = FER()
    cap = cv2.VideoCapture(0)
    return emotion_detector, cap

def detect_emotions(frame, emotion_detector):
    """Detect emotions in a given frame."""
    return emotion_detector.detect_emotions(frame)

def update_current_emotions(emotions, current_emotions, last_update_time, update_interval):
    """Update the current emotions based on detected emotions."""
    current_time = time.time()
    if current_time - last_update_time > update_interval:
        if emotions:
            current_emotions.update(emotions[0]['emotions'])  # Use the first detected face
            last_update_time = current_time
    return current_emotions, last_update_time

def draw_bounding_boxes(frame, emotions):
    """Draw bounding boxes around detected faces in the frame."""
    for emotion in emotions:
        bbox = emotion['box']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

def prepare_emotion_texts(current_emotions):
    """Prepare text to display current emotions."""
    emotion_texts = []
    for e, score in current_emotions.items():
        if score > 0.1:  # Only show emotions with confidence > 10%
            emotion_texts.append(f"{e}: {score*100:.1f}%")
    return emotion_texts

def display_emotions(frame, emotion_texts):
    """Display the current emotions on the frame."""
    y_offset = 30
    for text in emotion_texts:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 15

def check_emotion_alerts(current_emotions, frame, y_offset):
    """Check for specific emotions and trigger alerts."""
    if current_emotions.get('angry', 0) > 0.3:  # Lowered threshold for testing
        cv2.putText(frame, "Alert: Anger Detected!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif current_emotions.get('sad', 0) > 0.3:  # Lowered threshold for testing
        cv2.putText(frame, "Alert: Sadness Detected!", (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def main():
    """Main function to run the emotion detection program."""
    emotion_detector, cap = initialize_emotion_detector()
    emotion_history = deque(maxlen=10)
    last_update_time = time.time()
    update_interval = 15  # seconds
    current_emotions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        emotions = detect_emotions(frame, emotion_detector)
        current_emotions, last_update_time = update_current_emotions(emotions, current_emotions, last_update_time, update_interval)

        draw_bounding_boxes(frame, emotions)
        emotion_texts = prepare_emotion_texts(current_emotions)
        display_emotions(frame, emotion_texts)
        
        # Check for specific emotion alerts
        check_emotion_alerts(current_emotions, frame, 30)

        # Display the frame
        cv2.imshow('Emotion Recognition', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
