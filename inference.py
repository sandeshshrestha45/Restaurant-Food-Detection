# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 18:50:11 2021

@author: sandesh
"""

"""
├── yolo
│   ├── labels.txt
│   ├── yolov4-obj.cfg
│   ├── yolov4-obj_best.weights
├── examples
│   ├── apple_pie_before.jpg
│   ├── apple_pie_after.jpg
│   ├── chicken_curry_before.jpg 
│   ├── chicken_curry_after.jpg 
│   ├── french_fries_before.png
│   ├── french_fries_after.png
│   ├── fried_rice_before.jpg
│   ├── fried_rice_after.jpg
│   ├── before.jpg
│   ├── after.jpg
└── inference.py
 if program cant find yolo folder in main folder it will crash."""
 # example usage: python inference.py -i1 examples/before.jpg -i2 examples/after.jpg -o output.jpg
 
 
import argparse
import time
import glob

import cv2
import numpy as np

from skimage.measure import compare_ssim
import imutils


parser = argparse.ArgumentParser()
parser.add_argument("-i1", "--input1", type=str, default="",
	help="path to (optional) input image file")
parser.add_argument("-i2", "--input2", type=str, default="",
	help="path to (optional) input image file")
parser.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output image file. Write only the name, without extension.")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
	help="threshold for non maxima supression")

args = vars(parser.parse_args())

CONFIDENCE_THRESHOLD = args["confidence"]
NMS_THRESHOLD = args["threshold"]

impath1 = args["input1"]
impath2 = args["input2"]

imgA = cv2.imread(impath1)
imgB = cv2.imread(impath2)

# set a new height in pixels
new_height = 500
new_width = 750

# dsize
dsize = (new_width, new_height)

# resize image
imageA = cv2.resize(imgA, dsize, interpolation = cv2.INTER_AREA)
imageB = cv2.resize(imgB, dsize, interpolation = cv2.INTER_AREA)

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

mask = np.zeros(imageB.shape[:2], dtype="uint8")
largest_areas = sorted(cnts,key=cv2.contourArea)



# loop over the contours
for c in cnts:
    area = cv2.contourArea(c)
    if area > 4000:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        masked_img=cv2.drawContours(mask,[c],0,(255,255,255),-1)
        masked = cv2.bitwise_and(imageB, imageB, mask=mask)


cv2.imwrite("result.png", masked)
cv2.waitKey(0)
time.sleep(2)

impath="result.png"
weights = glob.glob("yolo/yolov4-obj_best.weights")[0]
labels = glob.glob("yolo/labels.txt")[0]
cfg = glob.glob("yolo/yolov4-obj.cfg")[0]

print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

lbls = list()
with open(labels, "r") as f:
	lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect(imgpath, nn):
	image = cv2.imread(imgpath)
	(H, W) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
	nn.setInput(blob)
	start_time = time.time()
	layer_outs = nn.forward(layer)
	end_time = time.time()

	boxes = list()
	confidences = list()
	class_ids = list()

	for output in layer_outs:
		for detection in output:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]

			if confidence > CONFIDENCE_THRESHOLD:
				box = detection[0:4] * np.array([W, H, W, H])
				(center_x, center_y, width, height) = box.astype("int")

				x = int(center_x - (width / 2))
				y = int(center_y - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_ids.append(class_id)


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[class_ids[i]]]
			cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
			text = "{}: {:.4f}".format(lbls[class_ids[i]], confidences[i])
			cv2.putText(image, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			label = "Inference Time: {:.2f} ms".format(end_time - start_time)
			cv2.putText(image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
			print(class_ids[i])
			print(lbls[class_ids[i]])


	cv2.imshow("image", image)
	if args["output"] != "":
		cv2.imwrite(args["output"], image)
	cv2.waitKey(0)

detect(impath, net)
