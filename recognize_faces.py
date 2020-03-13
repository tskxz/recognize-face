# Import the open cv model
import cv2

# Cascade to detect the frontal face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loads the image
img = cv2.imread("me_peoples.jpg")

# Converts to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.4, minNeighbors=5)
print(faces)
print(type(faces))

# Design the square detector faces
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 8)

# Show the image on a window
resized_image = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
cv2.imshow("Face detector", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindws()
