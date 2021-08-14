# YOLO_object_recognition
Python script for detecting objects and subtracting background.

**Requirements:**
1. Python3
2. [CV2](https://opencv.org/)
3. Numpy
4. [YOLOv3](https://pjreddie.com/darknet/yolo/) 

### Step 1
Make sure YOLOv3 weights, config file and names are saved in the right directory. In the script, make a connection between openCV and the readNet. 

```python
weights = "yolo-coco/yolov3.weights"
cfg = "yolo-coco/yolov3.cfg"
coco = "yolo-coco/coco.names"
net = cv2.dnn.readNet(weights, cfg)
classes = []
with open(coco, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] 
```
The `output_layers` variable will be called after the blobFromImage is put into the readNet:
```python
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
```
Next lines are to store the location, information and confidence of the detected objects:
```python
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
```
Based on this, the indexes can be made:
```python
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```
### Step 2

Next, the visualizing can be done:

```python
for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cvtImg = img[y:y+h,x:x+w]
```
With `cvtImg = img[y:y+h,x:x+w]`, a section of the captured frame is cropped, and then checked if it contains data:
```python
if np.shape(cvtImg) == ():
                    print("Image is empty ()")
                    break
                elif np.all(cvtImg == None):
                    print(np.all(cvtImg == None))
                    break
 ```
 This is necessary, because Numpy can throw an error if it has to operate on an non-existent/empty variable.
 
 For this code, if the data contains a Person, try to inverse the skincolor:
 ```python
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
                    except:
                        print("Couldn't load person image")
                        break
```
Here, the `cv2.bitwise_and()` operator works very nice to mask out colours.

### Step 3
Output the images as cascading widnows:
```python
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
```
First, resize the ouput image from the object detection, create a windowsName. 
Use `cv2.setWindowProperty(windowsName, cv2.WND_PROP_TOPMOST, 1)` to set the new window on top of the previous ones.
Move window to the edge of the previous window, so it moves to the right. If the screenwidth is reached, start at the left.
Because each window gets a unique name, according to the labels and added numbers, most windows will remain for some time until new images are created with the same window name.

 
                   
        


