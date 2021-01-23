import requests 

allimagesurl = []

def imagesurl():
   ImgUrl = "http://pictick.itfuturz.com/";
   r = requests.get('http://pictick.itfuturz.com/api/AppAPI/GetAlbumPhotoList?AlbumId=47') 
   for i in range(0,len(r.json()["Data"][0]["PhotoList"])):
       allimagesurl.append(ImgUrl + r.json()["Data"][0]["PhotoList"][i]["Photo"])
   return allimagesurl

