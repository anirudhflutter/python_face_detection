import cv2
import dlib
import argparse
import time
import math
import face_recognition
import os
from skimage import io
import getalldatafromapi
import urllib.request
from pil import Image 
H = 64
W = 64
import numpy as np
import url_to_image

# def extractFaces(path):
#   try:  
#     image = face_recognition.load_image_file(path)
#     hog_face_detector = dlib.get_frontal_face_detector()
#     faces_hog = hog_face_detector(image, 1)
#     end = time.time()
#     for face in faces_hog:
#       x = face.left()
#       y = face.top()
#       w = face.right() - x
#       h = face.bottom() - y

#       radius = np.round((math.sqrt((w*w) + (h*h))//2)).astype("int")

#     #   cv2.circle(image, (x+(w//2),y+(h//2)),radius , (255, 0, 0) , 1)
#     # draw box over face
#       topcoordinate = np.round((y + (h//2)) + (math.sqrt((w*w) + (h*h)))//2).astype("int")
#       bottomcoordinate = np.round((y + (h//2)) - (math.sqrt((w*w) + (h*h)))//2).astype("int")
#       leftcoordinate = np.round((x + (w//2)) - (math.sqrt((w*w) + (h*h)))//2).astype("int")
#       rightcoordinate = np.round((x + (w//2)) + (math.sqrt((w*w) + (h*h)))//2).astype("int")
    
#     #   roi_color = image[bottomcoordinate:topcoordinate, leftcoordinate:rightcoordinate]
#       roi_color = image[y-10:y+h+10, x-10:x+h+10]

#       cv2.imwrite("hog model/allfaces/{}".format(path),roi_color)
#   except Exception as e:
#     print(str(e))
    #   cv2.circle(image, (x+(w//2),y+(h//2)),radius+1 , (255, 0, 0) , 1)

uniquefaces = []

def hogmodel(imageurl,count): 
    image = io.imread(imageurl)
    if image is None:
     print("Could not read input image")
     exit()
    
# initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

# initialize cnn based face detector with the weights
    start = time.time()

# apply face detection (hog)
    faces_hog = hog_face_detector(image, 1)

    end = time.time()
    print("Execution Time (in seconds) :")
    print("HOG : ", format(end - start, '.2f'))

# loop over detected faces
    for face in faces_hog:
      x = face.left()
      y = face.top()
      w = face.right() - x
      h = face.bottom() - y
      
      givenface = image[y-10:y+h+10, x-10:x+h+10]
      givenfaceImage = Image.fromarray(givenface)
      givenfaceImage.show()
      if(len(uniquefaces) == 0):
        uniquefaces.append(givenfaceImage)
      else:
        uniquefaces.append(givenfaceImage)
        for i in uniquefaces:
          try:   
            known_image = face_recognition.load_image_file(givenfaceImage) 
            unknown_image = face_recognition.load_image_file(i)
            biden_encoding = face_recognition.face_encodings(known_image)[0] 
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0] 
            results = face_recognition.compare_faces([biden_encoding], unknown_encoding,tolerance=0.6) 
            if(results[0] == True):
               cv2.imwrite(("hog model/uniquefaces/{}".format("asd")), image)
          except Exception as e:
              print(str(e))

    # draw box over face
      # radius = np.round((math.sqrt((w*w) + (h*h))//2)).astype("int")

    #   cv2.circle(image, (x+(w//2),y+(h//2)),radius+1 , (255, 0, 0) , 1)

# write at the top left corner of the image
# for color identification
#     img_height, img_width = image.shape[:2]
#     cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (0,255,0), 2)

# # display output image
#     cv2.imwrite(("hog model/circularimages/{}".format(filename)), image)


count = 0
if __name__ == "__main__":

    alldata = getalldatafromapi.imagesurl()
    # data = os.listdir('images/')
    # for i in data:
    #     count+=1
    #     hogmodel(i,count)
    for i in range(0,len(alldata)):
        hogmodel(alldata[i],count)
        break
        
