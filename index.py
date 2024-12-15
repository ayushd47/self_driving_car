import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import cloud_integration
import distance_measurement
print(distance_measurement.measure_distance())

# from cloud_integration 
# GPIO Pins
TRIG = 23
ECHO = 24
LEFT_SIGNAL = 27
RIGHT_SIGNAL = 22
MOTOR_PIN = 17

measure_distance()

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(LEFT_SIGNAL, GPIO.OUT)
GPIO.setup(RIGHT_SIGNAL, GPIO.OUT)
GPIO.setup(MOTOR_PIN, GPIO.OUT)

# Firebase Setup
FIREBASE_URL = "https://autonomous-vehicle-b897a-default-rtdb.firebaseio.com/"
db = firebase.FirebaseApplication(FIREBASE_URL, None)

# Lane Detection Function
def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

# Pedestrian and Traffic Light Detection
def detect_objects(frame, net, classes):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["person", "traffic light"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

# Save data to the cloud
def save_to_cloud(speed, distance, total_distance):
    data = {
        "speed": speed,
        "distance_to_obstacle": distance,
        "total_distance_traveled": total_distance,
    }
    db.post("/car-data", data)

# Main Function
def main():
    total_distance = 0
    cap = cv2.VideoCapture(0)
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Measure distance
            distance = measure_distance()
            if distance < 30:
                GPIO.output(MOTOR_PIN, False)
                print("Obstacle detected! Stopping car.")
                save_to_cloud(0, distance, total_distance)
            else:
                GPIO.output(MOTOR_PIN, True)

            # Detect lane and objects
            frame = detect_lane(frame)
            frame = detect_objects(frame, net, classes)

            # Display the frame
            cv2.imshow("Car Camera", frame)

            # Update cloud data
            total_distance += 0.01  # Simulated distance increment
            save_to_cloud(50, distance, total_distance)  # Assuming constant speed 50 km/h

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
