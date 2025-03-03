from flask import request,jsonify
from database import Session
from models.UserPreference import UserPreference

def save_preferences():
    session = Session()
    try:
        data=request.get_json()
        user_id=data.get('user_id')
        peripheral_threshold = data.get('peripheral_threshold')
        distance_threshold = data.get('distance_threshold')
        distance_status = data.get('distance_status')
        peripheral_status = data.get('peripheral_status')
        color_status = data.get('color_status')
        if not user_id:
            return jsonify({'Error':'Please provide valid user_id'}),409
        if peripheral_threshold is None:
            peripheral_threshold=10
        if distance_threshold is None:
            distance_threshold = 10
        user=session.query(UserPreference).filter(UserPreference.user_id==user_id).first()
        if user:
            return jsonify({'Error':'User Already exist'}),409
        preference_data=UserPreference(user_id,peripheral_threshold,distance_threshold,distance_status,peripheral_status,color_status)

        session.add(preference_data)
        session.commit()
        return jsonify({'message':'Preferences saved successfully'}),200
    except Exception as e:
        session.rollback()
        return jsonify({'Error':str(e)}),500
    finally:
        session.close()

def get_preference(id):
    session = Session()
    try:
        pref_data=session.query(UserPreference).filter(UserPreference.user_id==id).first()
        if not pref_data:
            return jsonify({"Error":"User does not exist"}),404
        data = {
            'pre_id': pref_data.pre_id,
            'user_id': pref_data.user_id,
            'peripheral_threshold': pref_data.peripheral_threshold,
            'distance_threshold': pref_data.distance_threshold,
            'distance_status': pref_data.distance_status,
            'peripheral_status': pref_data.peripheral_status,
            'color_status': pref_data.color_status
        }
        return jsonify({"message":data}),200
    except Exception as e:
        session.rollback()
        return jsonify({'Error': str(e)}), 500
    finally:
        session.close()

def update_preferences(id):
    session = Session()
    try:
        data=request.get_json()

        peripheral_threshold = data.get('peripheral_threshold')
        distance_threshold = data.get('distance_threshold')
        distance_status = data.get('distance_status')
        peripheral_status = data.get('peripheral_status')
        color_status = data.get('color_status')
        user_data=session.query(UserPreference).filter(UserPreference.user_id==id).first()
        if not user_data:
            return jsonify({"message":"User id doesnot exist"}),201
        user_data.user_id=id
        user_data.peripheral_threshold=peripheral_threshold
        user_data.distance_threshold=distance_threshold
        user_data.distance_status=distance_status
        user_data.peripheral_status=peripheral_status
        user_data.color_status=color_status
        session.commit()
        return jsonify({"message":"User Preferences updated successfully"}),200
    except Exception as e:
        session.rollback()
        return jsonify({'Error------': str(e)}), 500
    finally:
        session.close()