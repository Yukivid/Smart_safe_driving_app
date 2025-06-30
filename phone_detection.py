import cv2
import numpy as np
import pyttsx3

def voice_notification():
    """
    Use text-to-speech to alert the driver not to use a phone while driving.
    """
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # female voice
    engine.setProperty('rate', 150)
    engine.say("Please don't use your phone while driving")
    engine.runAndWait()

def detect_phone_usage():
    """
    Detects phone usage using YOLO object detection.
    """
    net = cv2.dnn.readNet(r"C:\Users\hrith\Desktop\AI PROJECT\yolov3.weights", r"C:\Users\hrith\Desktop\AI PROJECT\yolov3 (1).cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(0)
    frame_id = 0
    cellphone_detected = False

    while True:
        ret, frame = cap.read()
        frame_id += 1

        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    if class_id == 67 and not cellphone_detected:  # '67' is typically the class ID for cell phones
                        cellphone_detected = True
                        voice_notification()

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0) if class_ids[i] != 67 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # quit on 'q' key press
            break

        cellphone_detected = False  # Reset the flag

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_phone_usage()
