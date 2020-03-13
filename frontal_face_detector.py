import cv2

# Load image
image = cv2.imread("me.jpg")

# Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Converts to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect a face
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
print(type(faces))
print(faces)
# Draw the rectangle
for x, y, w, h in faces:
    image = cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 3)
# Show the image in a new window
resized_image = cv2.resize(image, (int(image.shape[1] / 3), int(image.shape[0] / 3)))
cv2.imshow("Detector", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
