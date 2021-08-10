import cv2

#load some pre-trained data on face frontals from opencv (haar casacade algortihm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image
img = cv2.imread('img2.jpg')

#convert the chosen image to grayscale
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_img)

print(face_coordinates)

#display image
cv2.imshow('FD',gray_img)
cv2.waitKey()

print("End of program")

#testing git
#testing git 2
