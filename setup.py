import shutil
import os
if os.path.isdir("image"): 	
	shutil.rmtree("image") 
	
if os.path.isdir("image_test"):
	shutil.rmtree("image_test") 

import config
import collect_face_image
import collect_face_image_test
import train
import Real_Time