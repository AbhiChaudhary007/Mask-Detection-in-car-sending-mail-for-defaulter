import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import detect
import ocr_sql_mail
import os
import sys

path = os.getcwd()
weights = os.path.join(path,'yolov3', 'yolov3.weights')
cfg = os.path.join(path,'yolov3','yolov3.cfg')
def car(image):
    net = cv2.dnn.readNet(weights, cfg)
    model = load_model(os.path.join(path, 'yolov3','Face_Mask.h5'))

    classes = []

    with open(os.path.join(path,'yolov3', 'coco.names')) as f:
        classes = f.read().splitlines()

    img  = cv2.imread(image)
    height,width,channel = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop = False)

    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers)

    boxes = []
    confidences = []

    # 5,7
    #Find Car in Image
    for output in layerOutputs:
        for detection in output:
            confidence = detection[7]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x -w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes,confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))

    if indexes == ():
        print('No Car Detected')
        return
    else:
        for i in indexes.flatten():
            boxes_p = []
            confidences_p = []
            x, y, w, h = boxes[i]
            height,width,channel = img[y:y+h, x:x+w].shape
            blob = cv2.dnn.blobFromImage(img[y:y+h, x:x+w], 1/255.0, (416,416), swapRB=True, crop = False)
            net.setInput(blob)
            output_layers = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers)
            for output in layerOutputs:
                for detection in output:
                    confidence = detection[5]
                    if confidence >0.5 :
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w_p = int(detection[2]*width)
                        h_p = int(detection[3]*height)

                        x_p = int(center_x -w_p/2)
                        y_p = int(center_y - h_p/2)

                        boxes_p.append([x_p,y_p,w_p,h_p])
                        confidences_p.append(float(confidence))
            indexes_p = cv2.dnn.NMSBoxes(boxes_p,confidences_p, 0.5, 0.2)
            if indexes_p == ():
                print('No Person Detected')
                return
            colors = np.random.uniform(0,255,size=(len(boxes_p),3))
            for i in indexes_p.flatten():
                x_d, y_d, w_d, h_d = boxes_p[i]
                confidence = str(round(confidences_p[i],2))
                color = colors[i]
                person = img[y_d:y_d+h_d, x_d:x_d+w_d]
                person = cv2.resize(person, (224,224))
                preson = img_to_array(person)
                person = person.reshape(1,224,224,3)/255.0
                pred = model.predict(person)
                pred = pred.flatten()
                label = str(round(pred[0],2))
                if pred > 0.5 :
                    cv2.putText(img[y:y+h, x:x+w],label+ " Mask",(x_d,y_d),font, 1,(0,255,0),1)
                    cv2.rectangle(img[y:y+h, x:x+w],(x_d,y_d),(x_d+w_d,y_d+h_d),(0,255,0),2)
                else :
                    cv2.putText(img[y:y+h, x:x+w],label+ "  No Mask",(x_d,y_d), font, 1,(0,0,255),1)
                    cv2.rectangle(img[y:y+h, x:x+w],(x_d,y_d),(x_d+w_d,y_d+h_d),(0,0,255),2)
                    cv2.imwrite('crop.jpg',img[y:y+h,x:x+w])
                    source = 'crop.jpg'
                    detect.arg(source)
                    ocr_sql_mail.imgtext()

if __name__ == '__main__':
    image = sys.argv[2]
    car(image)
                
# cv2.imshow('Car', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()