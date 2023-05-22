import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import glob
import cv2
import numpy as np
import time
from datetime import datetime
from google.cloud import storage


#get info from firebase
cred = credentials.Certificate("Credentials_calcu-staging-firebase-adminsdk-j1o2c-51175324d6.json")
firebase_admin.initialize_app(cred)

db=firestore.client()

#load from GCP bucket
storage_client=storage.Client.from_service_account_json('Credentials_storage_credentials.json')

bucket = storage_client.get_bucket('calcu-bucket')


#model config paths
weights = glob.glob("yolov4-tiny-custom_final.weights")[0]
labels = glob.glob("labels.txt")[0]
cfg = glob.glob("yolov4-tiny-custom.cfg")[0]
print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))


#load model
lbls = []
with open(labels, "r") as f:
    lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i - 1] for i in net.getUnconnectedOutLayers()]

CONFIDENCE_THRESHOLD= 0.2
NMS_THRESHOLD= 0.4

IMG_WIDTH=416
IMG_HEIGHT=416


#get dates
print('From')
year1=int(input('Year'))
month1=int(input('Month'))
day1=int(input('Day'))
print('To')
year2=int(input('Year'))
month2=int(input('Month'))
day2=int(input('Day'))

start_dt = datetime(year1, month1, day1)
end_dt = datetime(year2, month2, day2)




def find_valid_list(start, end):
    result=db.collection("training_datasets").where("updated_at", ">", start).where("updated_at", "<", end).get()
    valid_list=[]
    for r in result:
        output=r.to_dict()
        valid_list.append(output)
    return valid_list

valid_list=find_valid_list(start_dt, end_dt)


def find_valid_entries(valid_list):
    doc_id=[] #document name
    eng_name=[]
    img_url=[]
    filename=[]
    for doc in valid_list:
        #print(doc['id']) 
        doc_id.append(doc['id'])
        img_url.append(doc['subtracted_img_url'])
        filename.append(doc['subtracted_img_url'].split("/")[-1].split(".")[0])
        #print(doc['updated_at'])
        for ing in doc['ingredients']:
            #print(ing['name_eng'])
            eng_name.append(ing['name_eng'])
    return doc_id, img_url,filename, eng_name


doc_id, img_url, filename, eng_name= find_valid_entries(valid_list)



def find_usable_filenames(filename):
    count=0
    usable_filenames_index=[]
    for elem in list(filename):
        if elem!='':
            usable_filenames_index.append(count)
        else: 
            filename.remove(elem)
        count=count+1
    return usable_filenames_index, filename


usable_filenames_index, filenames= find_usable_filenames(filename)



def find_usable_links(filenames):
    usable_links=[]
    for link in filenames:
        usable_links.append('output/'+link+'.jpg')
    return usable_links


usable_links=find_usable_links(filenames)




def detect(image, nn):
    #image = cv2.imread(imgpath)
    #image = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    nn.setInput(blob)
    start_time = time.time()
    layer_outs = nn.forward(layer)
    end_time = time.time()

    boxes = []
    confidences = []
    class_ids = []

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
        #print(bbcoordinates)
        return bbcoordinates

    #cv2.imshow("image", image)
    #if args["output"] != "":
        #cv2.imwrite(args["output"], image)
    #cv2.waitKey(0)


def find_bbox(usable_links):
    bbox_list=[]
    valid_bbox_index=[]
    for i in range(len(usable_links)):
        blob=bucket.blob(usable_links[i])
        img = np.asarray(bytearray(blob.download_as_bytes()), dtype=np.uint8)
        img = cv2.imdecode(img, flags=1)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        #print(img.shape,img)
        bbox=detect(img,net)
        #print(bbox)
        if bbox is not None:
            bbox_list.append(bbox)
            #print(i)
            valid_bbox_index.append(i)
    return bbox_list, valid_bbox_index



bbox_list, valid_bbox_index=find_bbox(usable_links)


valid_eng_name = list(map(eng_name.__getitem__, usable_filenames_index))
valid_eng_name_to_save = list(map(valid_eng_name.__getitem__, valid_bbox_index))



txt_file = open("predefined_classes_calcu1.txt", "r") #read class names from the predefined classes' file
ch='\n'
content_list = txt_file.readlines()

content_list  = list_of_str = [elem.replace(ch, '') for elem in content_list]
content_list  = list_of_str = [elem.replace(' ', '') for elem in content_list]



def find_ingredient_index(content_list, valid_eng_name_to_save):
    ing_index=[]
    for name in valid_eng_name_to_save:
        index=content_list.index(name)
        ing_index.append(index)
    return ing_index


ing_index=find_ingredient_index(content_list, valid_eng_name_to_save)


filename_to_save = list(map(filenames.__getitem__, valid_bbox_index))



def writefile(name,ind,a,b,c,d):
    outfile=open('{}.txt'.format(name),'w')
    outfile.write(ind+' ')
    outfile.write(a+' ')
    outfile.write(b+' ')
    outfile.write(c+' ')
    outfile.write(d)
    outfile.close()
    
    
    
image_w=333
image_h=250

i=0
while i<len(bbox_list):
    b=bbox_list[i][0]
    index=ing_index[i]
    file_name=filename_to_save[i]
    x1=b[0]
    y1=b[1]
    x2=b[2]
    y2=b[3]
    x_center = abs(((x2 + x1)/2)/image_w)
    y_center = abs(((y2 + y1)/2)/image_h)
    width = abs((x2 - x1)/image_w)
    height = abs((y2 - y1)/image_h)
    writefile(file_name,str(index),str(x_center),str(y_center),str(width),str(height))
    i=i+1




