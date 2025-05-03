from flask import request, jsonify
from models import User
from yolo.yolov8_08_front_end_img_process import detected_objects_from_front_end
from PIL import Image
import cv2
import numpy as np
from database import Session
from datetime import datetime
from models.UserLog import UserLog
import os
from sqlalchemy import desc


def front_end_frame_detect():
    session = Session()
    try:
        PROCESSED_IMAGES_DIR = "detected_images"
        if not os.path.exists(PROCESSED_IMAGES_DIR):
            os.makedirs(PROCESSED_IMAGES_DIR)

        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        user_id = request.form['user_id']
        camera_mode = request.form['camera_mode']
        user_id=int(user_id)
        camera_mode=int(camera_mode)

        # Validate user_id and camera_mode
        if not user_id or not camera_mode:
            return jsonify({"error": "user_id and camera_mode are required"}), 400

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        image = Image.open(file.stream)
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        processed_image, detected_data = detected_objects_from_front_end(img_np)

        if not detected_data:
            return jsonify({"message": "No objects detected."}), 200

        processed_image_filename = f"detected_image_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        processed_image_path = os.path.join(PROCESSED_IMAGES_DIR, processed_image_filename)
        cv2.imwrite(processed_image_path, processed_image)

        for obj in detected_data:
            if obj['detected_object'] == "textsignboard" or obj['detected_object'] == "speed":
                user_log = UserLog(
                    user_id=user_id,
                    detected_object=obj['detected_object'],
                    alert=obj["text"],
                    distance=obj['distance'],
                    date=datetime.now().date(),
                    time=datetime.now().time(),
                    img_path=processed_image_path,
                    camera_mode=camera_mode
                )
            elif obj['detected_object'] == "red":
                user_log = UserLog(
                    user_id=user_id,
                    detected_object=obj['detected_object'],
                    alert="Stop",
                    distance=obj['distance'],
                    date=datetime.now().date(),
                    time=datetime.now().time(),
                    img_path=processed_image_path,
                    camera_mode=camera_mode
                )
            else:
                user_log = UserLog(
                    user_id=user_id,
                    detected_object=obj['detected_object'],
                    alert="Detected an object",
                    distance=obj['distance'],
                    date=datetime.now().date(),
                    time=datetime.now().time(),
                    img_path=processed_image_path,
                    camera_mode=camera_mode
                )
            session.add(user_log)
        session.commit()
        session.close()

        processed_image_url = f'http://127.0.0.1:5000/detected_images/{os.path.basename(processed_image_path)}'

        return jsonify({
            "message": "Objects saved successfully!",
            "detected_objects": detected_data,
            "camera_mode": camera_mode,
            "processed_image_url": processed_image_url
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


def get_logs(id):
    session = Session()
    try:
        # Fetch logs for the given user_id
        user=session.query(User).filter(User.user_id==id).first()
        if not user:
            return jsonify({'Error': 'user does not exist'}), 409
        log_data = session.query(UserLog).filter(UserLog.user_id == id and User.user_id==id).all()

        # Check if logs exist for the user_id
        if not log_data:
            return jsonify({'Message': 'Log Does Not exist'}), 409

        # Prepare the response data
        data = []
        for index in log_data:
            data.append({
                'log_id': index.log_id,
                'user_id': index.user_id,
                'detected_object': index.detected_object,
                'alert': index.alert,
                'distance': index.distance,
                'date': index.date.isoformat() if index.date else None,
                'time': index.time.isoformat() if index.time else None,
                'img_path': f'http://192.168.1.2:5000/detected_images/{os.path.basename(index.img_path)}'
            })

        return jsonify({'message': data}), 200

    except Exception as e:
        return jsonify({'Error': str(e)}), 500
    finally:
        session.close()


def get_latest_undisplayed_log(user_id):
    session = Session()
    try:
        latest_log = session.query(UserLog).filter(
            UserLog.user_id == user_id,
            UserLog.is_displayed == False
        ).first()

        if latest_log:
            # Mark as displayed
            latest_log.is_displayed = True
            session.commit()
            session.refresh(latest_log)

            # Manually serialize to JSON
            log_data = {
                'log_id': latest_log.log_id,
                'user_id': latest_log.user_id,
                'detected_object': latest_log.detected_object,
                'alert': latest_log.alert,
                'distance': latest_log.distance,
                'date': latest_log.date.isoformat() if latest_log.date else None,
                'time': latest_log.time.isoformat() if latest_log.time else None,
                'img_path': latest_log.img_path,
                'camera_mode': latest_log.camera_mode,
                'is_displayed': latest_log.is_displayed
            }

            return jsonify({'message': log_data}), 200
        else:
            return jsonify({'message': "No data found"}), 201

    except Exception as e:
        return jsonify({'Error': str(e)}), 500
    finally:
        session.close()
