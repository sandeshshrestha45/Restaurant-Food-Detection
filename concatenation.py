import cv2
import numpy as np
import os
import random

# Concatenate images of different widths vertically
"""" A function to resize the image """
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

# Concatenate images of different heights horizontally
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

# Concatenate vertically and horizontally (like tiles)
def concat_tile_resize(img_list_2d, interpolation=cv2.INTER_CUBIC):
    img_list_v=[hconcat_resize_min(img_list_h, interpolation=cv2.INTER_CUBIC) for img_list_h in img_list_2d]
    return vconcat_resize_min(img_list_v, interpolation=cv2.INTER_CUBIC)

def Generate(): #function generates a random 6 digit number
	code = ''
	for i in range(6):
		code += str(random.randint(0,9))
	return code


for path in os.listdir("Yellow_Bell_Pepper"):
    if path.endswith('.jpg'):
        print(path)
        img1 = cv2.imread(f"Yellow_Bell_Pepper/{path}")
        img2 = img1
        img3 = img1
        height=img1.shape[0]
        width=img1.shape[1]
        if height>width and (width/height)<0.5:
            img = hconcat_resize_min([img1, img2, img3])
        elif height<width and (height/width)<0.5:
            img = vconcat_resize_min([img1, img2, img3])
        elif height<=100 and width<=100:
            img= concat_tile_resize([[img1,img2,img3],[img1,img2,img3],[img1,img2,img3]])
        else:
            img=img1
        dsize = (416, 416)
        output = cv2.resize(img, dsize)
        img_name= Generate() +'.jpg'
        folder="cropped_images"
        #cv2.imshow("Image", output)
        img_path = os.path.join(folder, img_name )
        # save image
        cv2.imwrite(img_path, output)
        #cv2.waitKey(0)

        
        