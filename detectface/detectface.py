from mtcnn.mtcnn import MTCNN
import cv2
import os
from PIL import Image

# Load MTCNN for face detection
mtcnn_detector = MTCNN()

#Resize img
input_path = 'detectface\Test5.jpg'
img = cv2.imread(input_path)
resized_img = cv2.resize(img, (675, 1200))

cv2.imshow('resize', resized_img)
cv2.imwrite("Test9.jpg", resized_img)

# Function to detect faces using MTCNN
def detect_faces_mtcnn(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    faces = mtcnn_detector.detect_faces(img_rgb)
    print(faces)

    # Check if any face is detected
    if faces:
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
            #print(x, y, width, height)

    # Display the result
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_faces_mtcnn("Test9.jpg")
