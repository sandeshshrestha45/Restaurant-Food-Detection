
 # example usage: python yolo_image.py -i street.jpg -o output.jpg
import argparse
import time
import glob
import cv2
import numpy as np
import csv
import os

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
	ids=[]
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
			bbcoordinates.append((x,y,w,h))
			ids.append(class_ids[i])
		#print(bbcoordinates)
		#print(ids)

	cv2.imshow("image", image)
	if args["output"] != "":
		cv2.imwrite(args["output"], image)
	cv2.waitKey(0)
	return image, bbcoordinates, ids
 
#image, bbox, class_num= detect(impath, net)


"""with open("data.csv", "a", newline="") as File:
	writer=csv.writer(File)
	writer.writerow([1,2,3,4,5])

File.close()
"""

def save_objects_to_csv(image,bbox, class_num):
	with open("data.csv", "a", newline="") as File:
		for i in range(len(bbox)):
			height, width, _ = image.shape
			classid= class_num[i]
			x,y,w,h= bbox[i]
			print(classid)
			print(x,y,w,h,width,height)
			writer=csv.writer(File)
			writer.writerow([classid,x,y,w,h,width,height])
	File.close()

#save_objects_to_csv(bbox, class_num)


for path in os.listdir("Yellow_bell_pepper"):
	#path = unicodedata.normalize("NFC", f"にんじん Carrot/{p}".strip())
	if path.endswith('.jpg'):
		print(path)
		image, bbox, class_num= detect(f"Yellow_bell_pepper/{path}", net)
		save_objects_to_csv(image, bbox, class_num)


