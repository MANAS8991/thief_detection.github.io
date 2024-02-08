import cv2
import numpy as np
import os

# Initialize the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the folder where the face photos will be stored
photos_path = 'path/to/store/photos'

# Create the photos folder if it doesn't exist
if not os.path.exists(photos_path):
    os.makedirs(photos_path)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam, or specify the index of the desired camera

# Variable to keep track of the number of captured photos
capture_count = 0

# Main loop for capturing and processing video frames
while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces and draw rectangles around them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Capture and save the photo when a face is detected
        photo = frame[y:y + h, x:x + w]
        photo_path = os.path.join(photos_path, 'face_{}.jpg'.format(len(os.listdir(photos_path))))
        cv2.imwrite(photo_path, photo)
        print('Face photo saved:', photo_path)

        # Increment the capture count
        capture_count += 1

        # Check if the capture count has reached the limit
        if capture_count >= 30:
            break

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit the loop if the capture count has reached the limit or 'q' is pressed
    if capture_count >= 30 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Check if there are photos for training
if len(os.listdir(photos_path)) == 0:
    print("No photos found in the specified folder for training.")
    exit()

# Prepare training data
faces = []
labels = []

# Iterate over the face photos
for filename in os.listdir(photos_path):
    if filename.endswith('.jpg'):
        image_path = os.path.join(photos_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = int(os.path.splitext(filename)[0].split('_')[1])
        faces.append(image)
        labels.append(label)

# Create the face recognizer using LBPH algorithm
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the face recognizer
face_recognizer.train(faces, np.array(labels))

# Convert the path into trained_model.xml file
model_path = os.path.join(photos_path, 'trained_model.xml')
face_recognizer.save(model_path)

print('Path converted to trained_model.xml file:', model_path)