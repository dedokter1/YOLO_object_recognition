# import numpy as np
# import cv2

# Load an color image in grayscale
# img = cv2.imread('image.png', 0)
# print(img)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

scrnWidth = 1920
scrnHeight = 1080
hue = 0
ycrcb_min = np.array((0,133,77), dtype='uint8')
ycrcb_max = np.array((155,173,127), dtype='uint8')
xpos = 0

weights = "yolo-coco/yolov3.weights"
cfg = "yolo-coco/yolov3.cfg"
coco = "yolo-coco/coco.names"
# Load Yolo
net = cv2.dnn.readNet(weights, cfg)
classes = []
with open(coco, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Capture
cap = cv2.VideoCapture(0)
print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print("Frame resolution set to: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")



while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        #print(int(frame.shape[0]))
        img = cv2.resize(frame, None, fx=0.5, fy=0.5)
        height, width, channels = img.shape
        #print(height,width)
        mask = np.zeros(img.shape[:2], dtype="uint8")
    # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

    # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Check for detections:

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                #color = colors[i]
                #cv2.rectangle(mask,(x, y), (x + w, y + h), 255, -1)
                #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(mask, label, (x, y - 5), font, 0.8, 255, 2)
                cvtImg = img[y:y+h,x:x+w]
                if np.shape(cvtImg) == ():
                    print("Image is empty ()")
                    break
                elif np.all(cvtImg == None):
                    print(np.all(cvtImg == None))
                    break
                elif label == "person":
                    try:
                        skinImg = cv2.cvtColor(cvtImg, cv2.COLOR_BGR2YCR_CB)
                        invImg = cv2.bitwise_not(skinImg)
                        mask = cv2.inRange(skinImg, ycrcb_min, ycrcb_max)
                        mask_inv = cv2.bitwise_not(mask)
                        invImg = cv2.bitwise_and(invImg, invImg, mask = mask_inv)
                        skinImg = cv2.bitwise_and(skinImg, skinImg, mask = mask)
                        output = cv2.add(invImg, skinImg)
                        cvtImg = cv2.cvtColor(output, cv2.COLOR_YCR_CB2BGR)
                        
                        #cvtImg = cv2.cvtColor(cvtImg, cv2.COLOR_BGR2HSV)
                        #h,s,v = cv2.split(cvtImg)
                        #hue = hue + 10
                        #hnew = np.mod(h + hue, 180).astype(np.uint8)
                        #hsv_new = cv2.merge([hnew,s,v])
                        #cvtImg = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
                    except:
                        print("Couldn't load person image")
                        break
                
                print(cvtImg.shape)
                cvtImg = cv2.resize(cvtImg, None, fx=3, fy=3)
                height, width, channels = cvtImg.shape
                #scrn = (i+1)%3
                scrn = i+1
                xpos = xpos + width
                if xpos > scrnWidth:
                    xpos = 0
                windowsName = label + "{:02d}".format(int(scrn))
                print(scrn, windowsName, xpos)
                cv2.namedWindow(windowsName, cv2.WINDOW_AUTOSIZE)
                cv2.setWindowProperty(windowsName, cv2.WND_PROP_TOPMOST, 1)
                cv2.moveWindow(windowsName, xpos, int((scrnHeight/2)-(height/2)))
                cv2.imshow(windowsName, cvtImg)
                cv2.waitKey(25)
                    #cv2.destroyWindow(windowsName)
                               
cap.release()    
cv2.destroyAllWindows()



