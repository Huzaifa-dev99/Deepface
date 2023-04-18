import cv2
import csv
from deepface import DeepFace

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Initialize the face counter and result dictionary
face_id = 0
face_results = {}

# Open a CSV file to write the results to
with open('face_results.csv', mode='w', newline='') as csv_file:
    fieldnames = ['face_id', 'gender', 'emotion', 'age']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through the frames of the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        # Loop through the faces
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            face_image = frame[y:y + h, x:x + w]

            # Detect the gender, emotion, and age of the face
            results = DeepFace.analyze(face_image, actions=['gender', 'emotion', 'age'], enforce_detection=False)

            # Loop through the results for each face detected
            for result in results:
                # Get the gender, emotion, and age from the result
                gender = result['gender']
                emotion = result['dominant_emotion']
                age = int(result['age'])

                # Draw a rectangle around the face and display the gender, emotion, and age
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame, f"{gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"{age}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check if this is a new face
                if face_id not in face_results:
                    # If so, increment the face counter and add the result to the dictionary
                    face_id += 1
                    face_results[face_id] = {'gender': gender, 'emotion': emotion, 'age': age}

                    # Write the result to the CSV file
                    writer.writerow({'face_id': face_id, 'gender': gender, 'emotion': emotion, 'age': age})

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # If the user presses the 'q' key, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
video_capture.release()

# Close the CSV file
csv_file.close()
