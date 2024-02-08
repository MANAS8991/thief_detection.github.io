import cv2
import numpy as np
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from playsound import playsound

# Initialize the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the folder where the face photos are stored
photos_path = 'path/to/store/photos'

# Path to the trained model
model_path = os.path.join(photos_path, 'trained_model.xml')

# Load the trained face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam, or specify the index of the desired camera

# Initialize the alarm sound
alarm_sound = 'Alarm10.wav'
# Set the alarm threshold (number of faces detected)
alarm_threshold = 1  # Adjust this value based on your requirements

# Email configuration
email_sender = 'girimanaskumar1998@gmail.com'  # Replace with your email address
email_password = 'Manas@9178'  # Replace with your email password
email_receiver = 'apukumargiri1@gmail.com'  # Replace with the recipient's email address

# Function to send email notification
def send_email():
    subject = 'Face Not Matched - Security Alert'
    body = 'A face has been detected, but it does not match any recognized labels.'

    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Create SMTP session and send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(email_sender, email_password)
        server.send_message(msg)

# Main loop for capturing and processing video frames
while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Flag to track if a face is detected
    face_detected = False

    # Iterate over detected faces and draw rectangles around them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Perform face recognition on the detected face
        face = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face)

        # Check if the face matches the recognized label
        if confidence < 70:  # Adjust the confidence threshold as needed
            # Face matched, display the label on the frame
            label_text = 'Match: {}'.format(label)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            label_text = 'No Match'
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            playsound(alarm_sound)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

# Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

