from flask import request, jsonify, session
from database import Session
from models.User import User

def password_change():
    session = Session()
    try:
        data=request.get_json()
        user_id = data.get('user_id')
        current_password=data.get('current_password')
        new_password=data.get('new_password')
        if current_password is None or new_password is None:
            return jsonify({'Error':'Please enter data'}),409
        if user_id is None:
            return jsonify({'Error':'Provide user id'}),409
        user=session.query(User).filter(user_id==User.user_id).first()
        if user is None:
            return jsonify({'Error':'User not found'}),409
        if user.password==current_password:
            user.password=new_password
            session.add(user)
            session.commit()
            return jsonify({'message':'Password changed successfully'}),200
        return jsonify({'Error':'Current password did not match'}),402
    except Exception as e:
        session.rollback()
        return jsonify({'Error':str(e)}),500
    finally:
        session.close()


def authenticate_user():
    session = Session()
    try:
        data=request.get_json()
        email=data.get('email')
        password=data.get('password')
        if not email or not password:
            return jsonify({'Error': 'Email and password both are required'}),400
        user=session.query(User).filter(email==User.email).first()
        if user and user.password==password:
            user.isLogOut=0
            session.commit()
            return jsonify({'message':'Login Successful','user_id':user.user_id}), 200
        else:
            return jsonify({'Error': 'Login Failed'}),401
    except Exception as e:
        return jsonify({'Error':str(e)}),500
    finally:
        session.close()


def password_reset():
    session = Session()
    try:
        data=request.get_json()
        email=data.get('email')
        if not email:
            return jsonify({'error':'Please provide an email'}),400
        user=session.query(User).filter(email==User.email).first()
        if user is None:
            return jsonify({'error': 'Email not registered'}), 400
        return jsonify({'message':'Password sent to '+email}),200
    except Exception as e:
        return jsonify({'error':str(e)}),500
    finally:
        session.close()


def new_user():
    session = Session()
    try:
        data=request.get_json()
        full_name=data.get('full_name')
        email= data.get('email')
        password = data.get('password')
        isLogOut = data.get('is_logout')
        isLogOut=int(isLogOut)
        if not full_name or not email or not password:
            return jsonify({'Error':'All fields are required'}),400
        is_valid_email=session.query(User).filter(email==User.email).first()
        if is_valid_email:
            return jsonify({'Error': 'Email already registered'}),409
        user=User(full_name,password,email,isLogOut)
        session.add(user)
        session.commit()
        return jsonify({'message':'User registered successfully'}),200
    except Exception as e:
        session.rollback()
        return jsonify({'Error': str(e)}),500
    finally:
        session.close()

def log_out(id):
    session = Session()
    try:
        user=session.query(User).filter(User.user_id==id).first()
        if not user:
            return jsonify({"error":"User doesnot exist"}),200
        user.isLogOut=1
        session.commit()
        return jsonify({"message":"User logged out successfully"}),200
    except Exception as e:
        session.rollback()
        return jsonify({'Error': str(e)}),500
    finally:
        session.close()