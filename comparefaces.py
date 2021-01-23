import face_recognition
import os

def comparefaces(firstimage,secondimage):
        try:   
            known_image = face_recognition.load_image_file(firstimage) 
            unknown_image = face_recognition.load_image_file(secondimage)
            biden_encoding = face_recognition.face_encodings(known_image)[0] 
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0] 
            results = face_recognition.compare_faces([biden_encoding], unknown_encoding) 
            print(results)
            return results[0]
        except Exception as e:
              print(str(e))

comparefaces("hog model/allfaces/_MG_9993.jpg","hog model/allfaces/_MG_9996.jpg")

allfiles= []

for files in os.listdir("hog model/allfaces/"):
    allfiles.append(files)

# for i in range(0,len(allfiles)):
#     for j in range(0,len(allfiles)):
#         if(comparefaces(allfiles[i],allfiles[j]) == True):
#             os.remove("{}".format(allfiles[j]))
#             allfiles.remove(allfiles[j])

for i in range(0,len(allfiles)):
    if(comparefaces(allfiles[i],"_MG_0001.jpg") == False):
            os.remove("{}".format(allfiles[i]))
            allfiles.remove(allfiles[i])

print(allfiles)