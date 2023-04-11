import cv2
from deepface import DeepFace

def detect_gender():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read the camera frame
        ret, frame = cap.read()

        # Analyze the image to detect faces and extract gender
        results = DeepFace.analyze(frame, actions=['gender'], enforce_detection=False)

        # Loop over the results and extract the gender for each face
        for result in results:
            gender = result['gender']

            # Convert gender to a string
            gender_str = str(gender)

            # Display the gender on the frame
            cv2.putText(frame, gender_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Gender Detection', frame)

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_gender()

