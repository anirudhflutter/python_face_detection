import cv2
import dlib
import argparse
import time
import numpy as np
import os
import face_recognition
import math  

def facedetection(filename):
    # load input image
   image = cv2.imread("images/{}".format(filename))

   if image is None:
     print("Could not read input image")
     exit()
    
# initialize hog + svm based face detector
   hog_face_detector = dlib.get_frontal_face_detector()
   
   start = time.time()

# apply face detection (hog)
   faces_hog = hog_face_detector(image, 1)
   print("faces_hog")
   print(faces_hog)
   end = time.time()
   print("Execution Time (in seconds) :")
   print("HOG : ", format(end - start, '.2f'))

# loop over detected faces
   for face in faces_hog:
       x = face.left()
       y = face.top()
       w = face.right() - x
       h = face.bottom() - y

    # draw box over face
       radius = np.round((math.sqrt((w*w) + (h*h))//2)).astype("int")

       cv2.circle(image, (x+(w//2),y+(h//2)),radius , (255, 0, 0) , 1)
      #  cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
       window_name = 'image'
       cv2.imshow(window_name, image) 
       roi_color = image[y:y + h, x:x + w]
       cv2.imwrite("allfaces/{}".format(str(w) + str(h) + '_faces.jpg'), roi_color)


   img_height, img_width = image.shape[:2]
   cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)

#    cv2.imwrite("hog model/"+"{}.png".format(filename), image)

def extractFaces(path):
    image = face_recognition.load_image_file("hog model/{}".format(path))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # faces = faceCascade.detectMultiScale(image, 
    # scaleFactor = 1.3, minNeighbors = 5
    # )
    face_locations = face_recognition.face_locations(image,model="hog")

    face_landmarks = face_recognition.face_landmarks(image)

    for (x, y, w, h) ,landmarks in zip(face_locations,face_landmarks):
            roi_color = image[y:y + h, x:x + w]
            print(roi_color)
            cv2.imwrite("allfaces/{}".format(str(w) + str(h) + '_faces.jpg'), roi_color)


if __name__ == "__main__":
    data = os.listdir('images/')
    for i in data:
        facedetection(i)

