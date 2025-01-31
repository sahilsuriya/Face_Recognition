import face_recognition
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for, Response

app = Flask(__name__)
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
    def load_known_faces(self):
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(os.path.splitext(filename)[0])
        print(f"Loaded {len(self.known_face_encodings)} known faces")

    def add_known_face(self, image_path, name):
        """Add a new face to the system, checking for duplicates."""
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        
        if not face_encoding:
            print("No faces found in the uploaded image.")
            return
        
        face_encoding = face_encoding[0]
        
        # Check if the encoding already exists in the known faces
        for known_encoding in self.known_face_encodings:
            matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
            if True in matches:
                print(f"Face already exists in the system: {name}")
                return  # Skip adding if the face is already known
        
        # Save the new face and encoding
        new_path = os.path.join(KNOWN_FACES_DIR, f"{name}{os.path.splitext(image_path)[1]}")
        #cv2.imwrite(new_path, cv2.imread(image_path))  # Save the image
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        print(f"Added {name} to known faces")

    
    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        return frame

def generate_frames(face_system):
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame = face_system.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

face_system = FaceRecognitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

from werkzeug.utils import secure_filename
import time

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files or 'name' not in request.form:
        return 'Missing image or name', 400
    
    image = request.files['image']
    name = request.form['name']
    
    if image.filename == '':
        return 'No selected file', 400
    
    # Use secure filename to avoid any security risks
    filename = secure_filename(image.filename)
    
    # Add a timestamp or unique identifier to the filename to prevent overwriting
    #timestamp = int(time.time())  # You can use a timestamp or a UUID
    unique_filename = f"{name}_{filename}"
    
    image_path = os.path.join(KNOWN_FACES_DIR, unique_filename)
    
    # Save the uploaded image with the new unique filename
    image.save(image_path)
    
    # Add the new face to the system
    face_system.add_known_face(image_path, name)
    
    return 'Face uploaded successfully', 200



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(face_system), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
