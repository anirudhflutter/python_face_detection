import cv2
import dlib
import argparse
import time
import os

def facedetection(filename):
   image = cv2.imread("images/{}".format(filename))

   if image is None:
     print("Could not read input image")
     exit()
     
   ap = argparse.ArgumentParser()
   args = ap.parse_args()
   cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

   start = time.time()

# apply face detection (cnn)
   faces_cnn = cnn_face_detector(image, 1)

   end = time.time()
   print("CNN : ", format(end - start, '.2f'))
   print(faces_cnn)

# loop over detected faces
   for face in faces_cnn:
     print("1")
     x = face.rect.left()
     y = face.rect.top()
     w = face.rect.right() - x
     h = face.rect.bottom() - y
     print("2")
     # draw box over face
     cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
     print("3")
     img_height, img_width = image.shape[:2]
    #  cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0,0,255), 2)
     print("4")
     roi_color = image[y:y + h, x:x + w]
     print("5")
     cv2.imwrite("CNN model/{}".format(str(w) + str(h) + '_faces.jpg'), roi_color)

if __name__ == "__main__":
    data = os.listdir('images/')
    for i in data:
        facedetection(i)

