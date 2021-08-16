import cv2

#load some pre-trained data on face frontals from opencv (haar casacade algortihm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image
#img = cv2.imread('img8.jpg')

#to capture video from webcam
webcam = cv2.VideoCapture(0)

#iterate over frames
while True:

    #### read the current frame
    successful_frame_read, frame= webcam.read()


    #convert the chosen image to grayscale
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_img)

    #draw rectangles around the faces
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)


    print(face_coordinates)

    #display image with faces selected
    cv2.imshow('FD',frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break;

print("End of program")

#testing git
#testing git 2
