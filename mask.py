import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import pickle
import time
import cv2
import os
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str, default="face_detection_model",
	help="path to OpenCV's deep learning face detector")
args = vars(ap.parse_args())

# load face detector model from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load mask detector model
print("[INFO] loading mask detector model 讀取口罩辨識模型...")
model_location = "/home/jetson/final/05300355.h5"
net_final = load_model(model_location)
# setting labels
cls_list = [ 'Mask Weared Incorrect','Weared Mask','NO MASK']

print("[INFO] starting video stream 開啟攝影機...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()	
   
# input a frame and it will return the prediction of with(out) mask or weared incorrect and the possibilty
def maskDetector(frame):
  x = frame
  x = np.expand_dims(x, axis = 0)
  pred = net_final.predict(x)[0]
  top_inds = pred.argsort()[::-1][:5]
  max=-1
  index=-1
  
  return pred[top_inds[0]], cls_list[top_inds[0]]
  
def makeTextAndRectangle(frame,x,y,h,w,textColor,text,bgColor,
                            font=cv2.FONT_HERSHEY_SIMPLEX,
							font_scale=.5,
							font_thickness=1,):
							
  text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
  text_w, text_h = text_size
  cv2.rectangle(frame,(x,y-25),(x+text_w,y),bgColor,-1)
  cv2.putText(frame,text,(x, y-10),font,font_scale,textColor,font_thickness)
  cv2.rectangle(frame,(x,y),(w,h),bgColor,2)

x=True

while(True):
  frame = vs.read()
  frame = imutils.resize(frame, width=600)
  (height, width) = frame.shape[:2]
  # initialise maskDetector
  if x==True:
    crop=cv2.resize(frame,(144,144))
    maskDetector(crop)
    x=False
  
  # construct a blob from the image
  imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)
	
  # apply OpenCV's deep learning-based face detector to localize
  # faces in the input image
  detector.setInput(imageBlob)
  detections = detector.forward()
  
  # loop over the detections
  for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
	
    if confidence > .5:
      box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
      (x,y,w,h)=box.astype('int')

    # to make crop range bigger for maskDetector
      rh = h
      rw = w
      by = int(y-rh)
      bx = int(x-rw)
      bh = int(y+h+rh)
      bw = int(x+w+rw)
      crop=frame[y:h,x:w]
    
      try:
        crop=cv2.resize(crop,(144,144))
        result,info=maskDetector(crop)
        result = round(float(result)*100,1)
	    # colour BGR
	    # weared incorrect colour yellow
        if info == cls_list[0]:
          textColor=(0,0,0)
          text=str(info)+' '+str(result)+'%'
          bgColor=(0,255,255)
          makeTextAndRectangle(frame,x,y,h,w,textColor,text,bgColor)
        # with mask colour green
        if info == cls_list[1]:
          textColor=(0,0,0)
          text=str(info)+' '+str(result)+'%'
          bgColor=(0,255,0)
          makeTextAndRectangle(frame,x,y,h,w,textColor,text,bgColor)
        # without mask colour red
        if info == cls_list[2]:
          textColor=(255,255,255)
          text=str(info)+' '+str(result)+'%'
          bgColor=(0,0,255)
          makeTextAndRectangle(frame,x,y,h,w,textColor,text,bgColor)
      except:
        pass
	
  cv2.imshow("Frame", frame)	
  
  #press q to escape
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  fps.update()
  #cv2.imshow("Frame", frame)

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
