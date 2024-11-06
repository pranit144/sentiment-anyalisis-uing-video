from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model  # Changed here
from tensorflow.keras.preprocessing.image import img_to_array  # Changed here
import cv2
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained face detection model and emotion classifier model
face_classifier = cv2.CascadeClassifier(r'D:\PycharmProjects\E-Learn-Frontend-Website-main\Video\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\PycharmProjects\E-Learn-Frontend-Website-main\Video\model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Video streaming generator function
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame into a format that can be streamed over HTTP
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in a byte format as part of a multi-part HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # A simple webpage to view the video stream

# Define a route to stream the video from the webcam
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Disable reloader to avoid issues with signal handling
