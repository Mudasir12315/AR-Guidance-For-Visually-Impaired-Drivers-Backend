from flask import Flask
from flask_cors import CORS
from controllers.UserLogController import front_end_frame_detect,get_logs
from controllers.UserController import authenticate_user,new_user,password_reset,password_change,log_out
from controllers.UserPreferenceController import save_preferences,get_preference,update_preferences
from models import create_tables
from flask import send_from_directory
from yolo.yolov8_08_front_end_img_process import load_models  # Add this import

create_tables()
print("Tables created successfully!")
app = Flask(__name__)
CORS(app)

# Load models when application starts
load_models()
print("Models loaded successfully!")


@app.route('/login',methods=['POST'])
def verify_user():
    result,status_code=authenticate_user()
    return result,status_code

@app.route('/signup',methods=['POST'])
def register_user():
    result,status_code=new_user()
    return result,status_code

@app.route('/reset/password',methods=['POST'])
def reset_password():
    result,status_code=password_reset()
    return result,status_code

@app.route('/change/password',methods=['PUT'])
def change_password():
    result,status_code=password_change()
    return result,status_code

@app.route('/logout/<int:id>',methods=['GET'])
def logOut(id):
    result,status_code=log_out(id)
    return result,status_code

@app.route('/user_log/<int:user_id>',methods=['GET'])
def user_log(user_id):
    result,status_code=get_logs(user_id)
    return result,status_code

@app.route('/frontend/frames/detection', methods=['POST'])
def frontend_frames_detection():
    print("Request Received")
    res,status_code=front_end_frame_detect()
    return res,status_code

@app.route('/detected_images/<filename>', methods=['GET'])
def serve_processed_image(filename):
    return send_from_directory('detected_images', filename)

@app.route('/get_preferences/<int:id>', methods=['GET'])
def get_preferences(id):
    res,status_code=get_preference(id)
    return res,status_code

@app.route('/save/preferences',methods=['POST'])
def save_pref():
    result,status_code=save_preferences()
    return result,status_code

@app.route('/update/preferences/<int:id>',methods=['PUT'])
def update_user_pref(id):
    result,status_code=update_preferences(id)
    return result,status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)