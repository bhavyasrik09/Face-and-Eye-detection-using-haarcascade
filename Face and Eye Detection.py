import cv2

# Load the face and eye classifiers
face_classifier = cv2.CascadeClassifier("C:\\Users\\user\\Desktop\\Edgematrix\\xml files\\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("C:\\Users\\user\\Desktop\\Edgematrix\\xml files\\haarcascade_eye.xml")

# Initialize video capture
video_capture = cv2.VideoCapture("C:\\Users\\user\\Desktop\\Edgematrix\\py files\\facedetection.mp4")

# Function to detect bounding boxes for faces and eyes
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Detect eyes within the detected face region
        eyes = eye_classifier.detectMultiScale(gray_image[y:y+h, x:x+w], 1.1, 5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(vid, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    return faces

# Main loop for video capture and detection
while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    # Detect faces and eyes
    faces = detect_bounding_box(video_frame)

    # Display the video frame with bounding boxes
    cv2.imshow("My Eye Detection Project", video_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
