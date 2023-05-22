
 # example usage: python yolo_image.py -i street.jpg -o output.jpg
import argparse
import time
import glob
import cv2
import numpy as np
import csv
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
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
impath = args["input"]

weights = glob.glob("yolo/yolov4.weights")[0]
labels = glob.glob("yolo/labels_calcu1.txt")[0]
cfg = glob.glob("yolo/yolov4-obj_calcu1.cfg")[0]

print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

lbls = list()
with open(labels, "r") as f:
	lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]

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
	bbcoordinates=[]
	names=[]
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[class_ids[i]]]
			#cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
			text = "{}: {:.4f}".format(lbls[class_ids[i]], confidences[i])
			#cv2.putText(image, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			label = "Inference Time: {:.2f} ms".format(end_time - start_time)
			#cv2.putText(image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
			bbcoordinates.append((x,y,w,h))
			names.append(lbls[class_ids[i]])
		print(bbcoordinates)
		print(names)

	#cv2.imshow("image", image)
	if args["output"] != "":
		cv2.imwrite(args["output"], image)
	cv2.waitKey(0)
	return image, bbcoordinates, names
 
#image, bbox, class_num= detect(impath, net)


def Generate(): #function generates a random 6 digit number
	code = ''
	for i in range(6):
		code += str(random.randint(0,9))
	return code


def crop_objects(image, bbox, classname):
	counts=dict()
	for i in range(len(bbox)):
		ing_name= classname[i]
		counts[ing_name] = counts.get(ing_name, 0) + 1
		x,y,w,h= bbox[i]
		#print(x,y,w,h)
		abs_x=abs(x)
		abs_y=abs(y)
		abs_w=abs(w)
		abs_h=abs(h)
		print(abs_x, abs_y,abs_w, abs_h)
		cropped_img = image[abs_y: abs_y+abs_h, abs_x: abs_x+abs_w]
		# construct image name and join it to path for saving crop properly
		#img_name = ing_name + '_' + str(counts[ing_name]) + '.jpg'
		img_name= Generate() + '_'+ str(counts[ing_name])+'.jpg'
		path="cropped_images"
		img_path = os.path.join(path, img_name )
		# save image
		cv2.imwrite(img_path, cropped_img)


for path in os.listdir("Yellow_bell_pepper"):
	#path = unicodedata.normalize("NFC", f"にんじん Carrot/{p}".strip())
	if path.endswith('.jpg'):
		print(path)
		image, bbox, classname= detect(f"Yellow_bell_pepper/{path}", net)
		crop_objects(image, bbox, classname)



