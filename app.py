import datetime
import os
from flask import Flask, redirect, url_for, request, jsonify, session, send_file
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
#from flask_dance.contrib.google import make_google_blueprint, google
from flask_mail import Mail, Message
from flask_jwt_extended import JWTManager, create_access_token, decode_token, get_jwt_identity, jwt_required
from flask_httpauth import HTTPBasicAuth
from dotenv import load_dotenv
from bson.objectid import ObjectId
import uuid
import jwt
from pymongo import TEXT, MongoClient
import secrets
from flask_cors import CORS

from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp 
import pickle
import base64

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = "deteksiyoga"
app.config['MONGO_URI'] = "mongodb://21090126:21090126@localhost:27017/21090126?authSource=auth"
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ardenarjuna28@gmail.com' #ganti pake email sendiri
app.config['MAIL_PASSWORD'] = 'qdwxvhgnfxokpzyk' 
app.config["JWT_SECRET_KEY"] = "deteksiyoga"
app.config['MAIL_DEFAULT_SENDER'] = 'ardenarjuna28@gmail.com' #ganti pake email sendiri

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
mail = Mail(app)
jwt = JWTManager(app)
auth = HTTPBasicAuth()
login_manager = LoginManager(app)
login_manager.login_view = 'login'
CORS(app)

client = MongoClient('mongodb://localhost:27017/')
db=client['21090126']
collection = db['users']


class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])  # Convert ObjectId to string
        self.username = user_data['username']
        self.email = user_data['email']
        self.is_verified = user_data.get('is_verified', False)
        self.api_key = user_data.get('api_key')

    @staticmethod
    def create_user(username, email, password=None, google_id=None):
        user = {
            "username": username,
            "email": email,
            "password": bcrypt.generate_password_hash(password).decode('utf-8') if password else None,
            "google_id": google_id,
            "is_verified": False,
            "api_key": secrets.token_hex(16)
        }
        result = collection.insert_one(user)
        user['_id'] = str(result.inserted_id)  # Convert ObjectId to string
        return user

    @staticmethod
    def find_by_email(email):
        return collection.find_one({"email": email})

    @staticmethod
    def find_by_google_id(google_id):
        return collection.find_one({"google_id": google_id})

    @staticmethod
    def verify_password(stored_password, provided_password):
        return bcrypt.check_password_hash(stored_password, provided_password)

    @staticmethod
    def set_verified(user_id):
        collection.update_one({'_id': ObjectId(user_id)}, {'$set': {'is_verified': True}})

    def update_password(self, new_password):
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        collection.update_one({'_id': ObjectId(self.id)}, {'$set': {'password': hashed_password}})

@login_manager.user_loader
def load_user(user_id):
    user = collection.find_one({"_id": ObjectId(user_id)})
    return User(user) if user else None

@auth.verify_password
def verify_password(email, password):
    user_data = User.find_by_email(email)
    if user_data and User.verify_password(user_data['password'], password):
        return User(user_data)
    return None

def verify_api_key(api_key):
    user_data = collection.find_one({"api_key": api_key})
    if user_data:
        return User(user_data)
    return None

def decodetoken(jwtToken):
    decode_result = decode_token(jwtToken)
    return decode_result


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"message": "Missing username, email, or password"}), 400

    existing_user = User.find_by_email(email)
    if existing_user:
        if existing_user.get('is_verified', False):
            return jsonify({"message": "Email already registered"}), 400
        else:
            # Resend verification email
            token = create_access_token(identity=str(existing_user['_id']), expires_delta=False)
            msg = Message('Email Verification', recipients=[email])
            msg.body = f'Your verification link is: {token}'
            mail.send(msg)
            return jsonify({"message": "Verification email sent. Please check your inbox."}), 200

    user_data = User.create_user(username=username, email=email, password=password)

    # Send verification email
    token = create_access_token(identity=user_data['_id'], expires_delta=False)
    msg = Message('Email Verification', recipients=[email])
    msg.body = f'Your verification link is: {token}'
    mail.send(msg)

    return jsonify({"message": "User registered successfully. Verification email sent."}), 201


# Define a text index on 'username' and 'email' fields for case-insensitive search
# collection.create_index([("username", TEXT), ("email", TEXT)], default_language='english')

@app.route('/bearer-auth', methods=['GET'])
@jwt_required()
def detail_user():
    bearer_auth = request.headers.get('Authorization', None)
    if not bearer_auth:
        return {"message": "Authorization header missing"}, 401

    try:
        jwt_token = bearer_auth.split()[1]
        token = decode_token(jwt_token)
        username = token.get('sub')

        if not username:
            return {"message": "Token payload is invalid"}, 401

        user = collection.find_one({"_id": ObjectId(username)})
        if not user:
            return {"message": "User not found"}, 404

        # Update is_verified to True
        collection.update_one({"_id": user["_id"]}, {"$set": {"is_verified": True}})

        data = {
            'username': user['username'],
            'email': user['email'],
            '_id': str(user['_id'])  # Convert ObjectId to string
        }
    except Exception as e:
        return {
            'message': f'Token is invalid. Please log in again! {str(e)}'
        }, 401

    return jsonify(data), 200

@app.route('/api/verify_email', methods=['POST'])
def verify_email():
    data = request.json
    code = data.get('code')

    if not code:
        return jsonify({"message": "Kode verifikasi tidak disediakan"}), 400

    try:
        # Decode the token (assuming token is used as verification code)
        decoded_token = jwt.decode(code, app.config['SECRET_KEY'], algorithms=["HS256"])
        user_email = decoded_token.get('user_email')

        if not user_email:
            return jsonify({"message": "Token tidak valid"}), 400

        user = mongo.db.users.find_one({"email": user_email})

        if not user:
            return jsonify({"message": "Pengguna tidak ditemukan"}), 404

        # Set user as verified
        User.set_verified(user['_id'])

        return jsonify({"message": "Verifikasi berhasil"}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Token telah kedaluwarsa"}), 400
    except jwt.InvalidTokenError:
        return jsonify({"message": "Token tidak valid"}), 400
    except Exception as e:
        return jsonify({"message": f"Kesalahan terjadi: {str(e)}"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    user_data = User.find_by_email(email)
    if user_data and User.verify_password(user_data['password'], password):
        if not user_data.get('is_verified'):
            return jsonify({"message": "Email not verified"}), 403
        user = User(user_data)
        login_user(user)
        return jsonify({"message": "Login successful"}), 200
    return jsonify({"message": "Invalid credentials"}), 401


@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logout successful"}), 200

@app.route('/api/reset_password', methods=['POST'])
def reset_password():
    data = request.json
    email = data.get('email')

    # Check if email exists in MongoDB
    user = mongo.db.users.find_one({'email': email})
    if not user:
        return jsonify({'message': 'Email not found'}), 404

    # Generate a token (could use JWT or other methods)
    # Send reset password email with the token
    
    return jsonify({'message': 'Reset password email sent'}), 200

@app.route('/change_email', methods=['POST'])
@login_required
def change_email():
    try:
        data = request.json
        new_email = data.get('new_email')

        if not new_email:
            return jsonify({"message": "Missing new email"}), 400

        user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user_data:
            return jsonify({"message": "User not found"}), 404

        # Send email confirmation
        token = create_access_token(identity=str(current_user.id), expires_delta=False)
        msg = Message('Email Change Confirmation', recipients=[new_email])
        msg.body = f'Your email change confirmation token is: {token}'
        mail.send(msg)

        return jsonify({"message": "Email change confirmation sent. Please check your inbox."}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500
    
@app.route('/confirm_change_email', methods=['POST'])
def confirm_change_email():
    bearer_auth = request.headers.get('Authorization', None)
    if not bearer_auth:
        return {"message": "Authorization header missing"}, 401

    try:
        jwt_token = bearer_auth.split()[1]
        token = decode_token(jwt_token)
        user_id = token.get('sub')

        if not user_id:
            return {"message": "Token payload is invalid"}, 401

        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"message": "User not found"}), 404

        data = request.json
        new_email = data.get('new_email')

        if not new_email:
            return jsonify({"message": "New email not provided"}), 400

        mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"email": new_email}})
        return jsonify({"message": "Email changed successfully"}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500
    
@app.route('/update_password', methods=['POST'])
@login_required
def update_password():
    try:
        data = request.json
        current_password = data.get('Password lama')
        new_password = data.get('Password baru')

        if not current_password or not new_password:
            return jsonify({"message": "Missing current password or new password"}), 400

        user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user_data:
            return jsonify({"message": "User not found"}), 404

        if not User.verify_password(user_data['password'], current_password):
            return jsonify({"message": "Current password is incorrect"}), 401

        current_user.update_password(new_password)
        return jsonify({"message": "Password updated successfully"}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/edit_profile', methods=['POST'])
@login_required
def edit_profile():
    try:
        data = request.form
        username = data.get('username')
        photo = request.files.get('photo')

        if not username:
            return jsonify({"message": "Missing username"}), 400

        user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user_data:
            return jsonify({"message": "User not found"}), 404

        update_data = {
            'username': username
        }

        if photo:
            projectPath = "/home/student/21090126/yogapose/"
            photo_filename = f"{current_user.id}.jpg"
            photo.save(os.path.join(projectPath, 'static/uploads', photo_filename))
            update_data['photo'] = photo_filename

        mongo.db.users.update_one({'_id': ObjectId(current_user.id)}, {'$set': update_data})

        return jsonify({"message": "Profile updated successfully"}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/user/image/<image_name>', methods=['GET'])
def get_image(image_name):
    return send_file(f"./static/uploads/{image_name}", mimetype="image/jpeg")

@app.route('/')
def home():
    return 'Hello World!'

socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(message):
    print('Received message:', message)


# Load Model
with open('assets\model_cobadataset.pkl', 'rb') as f:
    model = pickle.load(f)

# Mediapipe    
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

@socketio.on('image')
def handle_image(image_data):
    try:
        body_language_prob = 0.0
        body_language_class = "none"
        image_data_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_data_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (340, 180), interpolation=cv2.INTER_LINEAR)
            
            # Make Detections
            results = holistic.process(image)
            print('Detection results obtained')

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            try:
                if results.pose_landmarks is not None:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Concatenate rows
                    row = pose_row
                    
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    
                    print(f'class: {body_language_class}, prob: {body_language_prob}')

                else:
                    print('No pose landmarks detected')
                
            except Exception as e:
                print('Error during prediction:', e)
    
        processed_image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        processed_image_data = base64.b64encode(processed_image_bytes).decode('utf-8')
        prob_float = float(np.max(body_language_prob))
        prob = str(prob_float)
        print(prob)

        emit('response', {"imageData": processed_image_data, "pose_class": body_language_class, "prob": prob})

    except Exception as e:
        print('Error processing image:', e)

# MongoDB setup
client = MongoClient("mongodb+srv://sulapsempurna:sempurna0011@yoga.re1gige.mongodb.net/")
db = client["v4yoga"]
collection = db["deteksi"]

@app.route('/receivedata', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid data"}), 400

        tanggal = data.get("tanggal")
        pose_class = data.get("class")
        probability = data.get("probability")   

        if not all([tanggal, pose_class, probability]):
            return jsonify({"error": "Missing data fields"}), 400

        detection = {
            "tanggal": tanggal,
            "class": pose_class,
            "probability": probability
        }

        collection.insert_one(detection)
        return jsonify({"message": "Data successfully saved"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='194.31.53.102', port=21128)