from ultralytics import YOLO
import cv2
import os
import numpy as np
import easyocr
know_width_cls={
    "laptop":0.3302,  # for our testing purpose
    "car":1.7,
    "bike":0.7,
    "truck":2.4,
    "bus":2.5,
    "unknown":0
}

def detected_objects_from_front_end(image):
    # Load camera calibration matrix
    matrix = np.load('./calibration_matrix/camera_matrix.npy')
    focal_length_pixels = matrix[0, 0]

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Load YOLO model
    if not hasattr(detected_objects_from_front_end, "model"):
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, 'best_3.pt')
        detected_objects_from_front_end.model = YOLO(model_path)

    model = detected_objects_from_front_end.model
    detected_objects_list = []

    # Resize the image for YOLO processing
    image = cv2.resize(image, (640, 640))
#    cv2.imshow("recieved",image)
    results = model(image)

    if results and results[0] and results[0].boxes:
        for result in results[0].boxes:
            conf = result.conf[0]
            if conf >= 0.5:
                box = result.xyxy[0]
                cls = int(result.cls[0])
                x1, y1, x2, y2 = map(int, box)
                distance=0
                # Calculate distance
                if  model.names.get(cls, "unknown") in ["car","bus","truck","bike","unknown"]:
                    known_width=know_width_cls[model.names.get(cls,"unknown")]
                    perceived_width = x2 - x1
                    distance = (known_width * focal_length_pixels) / perceived_width
                    #distance = distance * 3.2808  # Convert meters to feet

                # Extract text from the specific region using EasyOCR
                text=""
                text_list=[]
                if model.names.get(cls,"unknown") == "textsignboard" or model.names.get(cls,"unknown") == "speed":
                    roi = image[y1:y2, x1:x2]
                    text_list = reader.readtext(roi,detail=0)
                    print(text_list)# Extract text without details
                for txt in text_list:
                    text=text+" "+txt
                detected_objects_list.append({
                    'detected_object': model.names.get(cls, "Unknown"),
                    'distance': distance,
                    'text': text if text else ""
                })

                # Draw bounding box and text on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f'{model.names.get(cls, "Unknown")} {distance:.2f} m',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image, detected_objects_list