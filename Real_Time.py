from keras.models import load_model
import glob
import cv2
import numpy as np
import os
import time

print("________________________LOADING...______________________")
names  = [] 
a = glob.glob("image/*")
b = " ".join(a).split(" ")
for i in b:
	names.append(i.split('\\')[1])

face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

def detect(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	faces = face_cascade.detectMultiScale(gray, 1.5,5)

	if faces is ():
		return None
	else:
		for x,y,w,h in faces:
			return (x,y,w,h)


model = load_model("test_vgg.h5")

os.system("clear")
print("\t\t\n________________________Starting Real Time....______________________\t\t\n")


cap = cv2.VideoCapture(0)
while True:

	ret, frame = cap.read()
	if detect(frame) is not None:

		x,y,w,h = detect(frame)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
		face = frame[y:y+h, x:x+w]
		face = cv2.resize(face, (224, 224))
		face_gray = face/255.0
		a= np.argmax(model.predict(face_gray.reshape(1,224,224,3)))

		x = names[a]
		face1 = cv2.resize(frame , (500,500))
	else:
		x = "Face is Not Found"
	
	cv2.putText(frame, str(x), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

	cv2.imshow("video", frame)
	
	

	if cv2.waitKey(1)==13:
	    break
cap.release()
cv2.destroyAllWindows()