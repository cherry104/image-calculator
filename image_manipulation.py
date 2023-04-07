import cv2
import numpy as np
# Load the input images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Convert the input images to grayscale for face detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Detect faces in the input images
faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5)
faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5)
# Extract facial features from each detected face in image 1
for (x, y, w, h) in faces1:
    roi_gray = gray1[y:y+h, x:x+w]
    roi_color = img1[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
# Merge the facial features from image 1 and image 2
for (x, y, w, h) in faces2:
    roi_gray = gray2[y:y+h, x:x+w]
    roi_color = img2[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
        img1[y+ey:y+ey+eh, x+ex:x+ex+ew] = eye_roi_color
# Display the output image
cv2.imshow('Output Image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
