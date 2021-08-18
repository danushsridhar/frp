import os
from PIL import Image
import numpy as np
import cv2
import pickle


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]
            print(label_ids)
            # y_lables.append(label)
            # x_lables.append(path)
            pil_image = Image.open(path).convert("L")  # grayscale and pil stands for python image library
            size = (1000,1000)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8") # this line convert each image into number, breaks down each pixel into numbers
            #print(image_array)
            faces = trained_face_data.detectMultiScale(image_array)

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
