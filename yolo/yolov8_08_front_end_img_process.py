import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO
import os

# Global variables to hold loaded models
model = None
reader = None


def load_models():
    global model, reader

    # Load YOLO model
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'best_5.pt')
    model = YOLO(model_path)  # Or your actual model path

    # Load EasyOCR reader
    reader = Reader(['en'], gpu=False)  # Adjust languages as needed

    # Warm up models with a dummy inference
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    model.predict(dummy_image)
    reader.readtext(dummy_image)

    print("Models loaded and warmed up")


know_width_cls={
    "laptop":0.3302,  # for our testing purpose
    "car":1.7,
    "bike":0.7,
    "truck":2.4,
    "bus":2.5,
    "unknown":0
}

def detected_objects_from_front_end(image):
    global model, reader
    # Load camera calibration matrix
    matrix = np.load('./calibration_matrix/camera_matrix.npy')
    focal_length_pixels = matrix[0, 0]

    detected_objects_list = []

    # Resize the image for YOLO processing
    image = cv2.resize(image, (640, 640))
    #cv2.imshow("recieved",image)
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


# import cv2
# import numpy as np
# from paddleocr import PaddleOCR
# from ultralytics import YOLO
# import os
# from skimage.transform import rotate
# from scipy.ndimage import interpolation as inter
#
# # Global variables to hold loaded models
# model = None
# ocr = None
#
#
# def preprocess_image(image):
#     """
#     Preprocess the ROI for better OCR accuracy.
#     """
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply CLAHE for contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)
#
#     # Denoise using Gaussian blur
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     # Adaptive thresholding to binarize
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
#
#     # Deskew the image
#     def correct_skew(image):
#         # Find angle using Hough transform
#         edges = cv2.Canny(image, 50, 150, apertureSize=3)
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#         angle = 0
#         if lines is not None:
#             for [[x1, y1, x2, y2]] in lines:
#                 angle += np.arctan2(y2 - y1, x2 - x1)
#             angle /= len(lines)
#             angle = angle * 180 / np.pi
#         # Rotate image to correct skew
#         rotated = rotate(image, angle, cval=255, preserve_range=True).astype(np.uint8)
#         return rotated
#
#     thresh = correct_skew(thresh)
#
#     # Resize for better OCR (optional, adjust scale factor as needed)
#     scale_factor = 1.5
#     thresh = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
#
#     return thresh
#
#
# def load_models():
#     global model, ocr
#     # Load YOLO model
#     script_dir = os.path.dirname(__file__)
#     model_path = os.path.join(script_dir, 'best_5.pt')
#     model = YOLO(model_path)
#
#     # Load PaddleOCR
#     ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)  # Set use_gpu=True if GPU is available
#
#     # Warm up YOLO model
#     dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
#     model.predict(dummy_image)
#
#     print("Models loaded and warmed up")
#
#
# know_width_cls = {
#     "laptop": 0.3302,
#     "car": 1.7,
#     "bike": 0.7,
#     "truck": 2.4,
#     "bus": 2.5,
#     "unknown": 0
# }
#
#
# def detected_objects_from_front_end(image):
#     global model, ocr
#     # Load camera calibration matrix
#     matrix = np.load('./calibration_matrix/camera_matrix.npy')
#     focal_length_pixels = matrix[0, 0]
#
#     detected_objects_list = []
#
#     # Resize the image for YOLO processing
#     image = cv2.resize(image, (640, 640))
#     results = model(image)
#
#     if results and results[0] and results[0].boxes:
#         for result in results[0].boxes:
#             conf = result.conf[0]
#             if conf >= 0.5:
#                 box = result.xyxy[0]
#                 cls = int(result.cls[0])
#                 x1, y1, x2, y2 = map(int, box)
#                 distance = 0
#
#                 # Calculate distance
#                 if model.names.get(cls, "unknown") in ["car", "bus", "truck", "bike", "unknown"]:
#                     known_width = know_width_cls[model.names.get(cls, "unknown")]
#                     perceived_width = x2 - x1
#                     distance = (known_width * focal_length_pixels) / perceived_width
#
#                 # Extract text from the specific region using PaddleOCR
#                 text = ""
#                 if model.names.get(cls, "unknown") in ["textsignboard", "speed"]:
#                     roi = image[y1:y2, x1:x2]
#                     # Preprocess the ROI
#                     processed_roi = preprocess_image(roi)
#                     # Save processed ROI temporarily for PaddleOCR
#                     temp_path = "temp_roi.png"
#                     cv2.imwrite(temp_path, processed_roi)
#                     # Perform OCR
#                     result = ocr.ocr(temp_path, cls=True)
#                     text_list = [line[1][0] for line in result[0]] if result and result[0] else []
#                     text = " ".join(text_list)
#                     print(f"Extracted text: {text}")
#                     # Clean up
#                     os.remove(temp_path)
#
#                 detected_objects_list.append({
#                     'detected_object': model.names.get(cls, "Unknown"),
#                     'distance': distance,
#                     'text': text if text else ""
#                 })
#
#                 # Draw bounding box and text on the image
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(image, f'{model.names.get(cls, "Unknown")} {distance:.2f} m',
#                             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     return image, detected_objects_list