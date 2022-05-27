from fileinput import filename
from genericpath import exists
from pathlib import Path
from tkinter import Pack
import easyocr
import statistics
import cv2
import numpy as np
import imutils
import time
import glob
import tensorflow as tf 
from PIL import Image
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use( 'tkagg' )
import cv2


# For obtaining the detections of the test_images, cropping and visualizing predictions
def get_predictions(IMAGE_PATH, PATH_TO_CKPT, PATH_TO_LABELS, detection_threshold):
    i = 0

    crop = input("Do you want to crop bounding boxes (yes/no): ")
    plot = input("Do you want to plot bounding boxes (yes/no): ")

    print('Loading model...', end='')
    start_time = time.time()

    detect_fn = tf.saved_model.load(PATH_TO_CKPT)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))


    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

    test_images_paths = glob.glob(IMAGE_PATH + "/*jpg")
    for test_image_path in test_images_paths:
        image = Image.open(test_image_path)
        image_np = np.array(image)
        file_name = os.path.basename(test_image_path)

        #print('Running inference for {}... '.format(test_image_path), end='')

        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        #input_tensor = input_tensor[:, :, :, :3]
        #input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)


        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


        image_np_with_detections = image_np.copy()


        if crop == 'yes':
            
            crop_objects(image, image_np_with_detections, detections, detection_threshold, file_name, i)

        if plot == 'yes':

            visualize_pcb(detections, image_np_with_detections, category_index, detection_threshold)


# script for created folders and getting bounding box images further used in OCR
def crop_objects(image, image_np_with_detections, detections, detection_threshold, file_name, i):

        
    Path("D:\\pcb_project\\{}".format(file_name)).mkdir(parents=True, exist_ok=True)
    Path("D:\\pcb_project\\{}\\resistors".format(file_name)).mkdir(parents=True, exist_ok=True)
    Path("D:\\pcb_project\\{}\\transistors".format(file_name)).mkdir(parents=True, exist_ok=True)
    Path("D:\\pcb_project\\{}\\capacitors".format(file_name)).mkdir(parents=True, exist_ok=True)
    Path("D:\\pcb_project\\{}\\ICs".format(file_name)).mkdir(parents=True, exist_ok=True)
    Path("D:\\pcb_project\\{}\\diodes".format(file_name)).mkdir(parents=True, exist_ok=True)
    Path("D:\\pcb_project\\{}\\inductors".format(file_name)).mkdir(parents=True, exist_ok=True)

    #global ymin, ymax, xmin, xmax
    width, height = image.size
    i=i
        #Coordinates of detected objects
    for detection in detections['detection_boxes']:
        ymin = int(detection[0]*height)
        xmin = int(detection[1]*width)
        ymax = int(detection[2]*height)
        xmax = int(detection[3]*width)
        crop_img = image_np_with_detections[ymin:ymax, xmin:xmax]
        

        if detections['detection_scores'][i] < detection_threshold:
            continue
        

        if detections['detection_classes'][i] == 1:
            cv2.imwrite('D:\\pcb_project\\{}\\resistors\\resistor'.format(file_name) +  str(i) +'.png', crop_img)
        elif detections['detection_classes'][i] == 2:
            cv2.imwrite('D:\\pcb_project\\{}\\transistors\\transistor'.format(file_name) +  str(i) +'.png', crop_img)
        elif detections['detection_classes'][i] == 3:
            cv2.imwrite('D:\\pcb_project\\{}\\capacitors\\capacitor'.format(file_name) +  str(i) +'.png', crop_img)
        elif detections['detection_classes'][i] == 4:
            cv2.imwrite('D:\\pcb_project\\{}\\ICs\\IC'.format(file_name) +  str(i) +'.png', crop_img)
        elif detections['detection_classes'][i] == 5:
            cv2.imwrite('D:\\pcb_project\\{}\\diodes\\diode'.format(file_name) +  str(i) +'.png', crop_img)
        elif detections['detection_classes'][i] == 6:
            cv2.imwrite('D:\\pcb_project\\{}\\inductors\\inductor'.format(file_name) +  str(i) +'.png', crop_img)

        i+=1


# For obtaining the OCR of the image
def ocr(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    #print(result)
    return result


#script for rotating image and saving image orientation with best accuracy 
def text_detection(img):
    result = ocr(img)
    

    list1 = []
    list2 = []
    list3 = []
    list4 = []


    for i in range(4):
        
        if i == 0:
            for accu in result:
                list1.append(accu[2])
            img1 = imutils.rotate_bound(img, 90)
            result1 = ocr(img1)
        if i == 1:
            for accu in result1:
                list2.append(accu[2])
            img2 = imutils.rotate_bound(img1, 90)
            result2 = ocr(img2)
        if i == 2:
            for accu in result2:
                list3.append(accu[2])
            img3 = imutils.rotate_bound(img2, 90)
            result3 = ocr(img3)
        if i ==3:
            for accu in result3:
                list4.append(accu[2])
                

    #print(list1, list2, list3, list4)

    best = best_results(list1, list2, list3, list4)


    if best==list1:
        visualize_ocr(img, result)
        return list1
    elif best==list2:
        visualize_ocr(img1, result1)
        return list1
    elif best==list3:
        visualize_ocr(img2, result2)
        return list1
    elif best==list4:
        visualize_ocr(img3, result3)
        return list1
        


# script for choosing best accuracy for different orientation
def best_results(list1, list2, list3, list4):
    a= max(list1)
    b= max(list2)
    c= max(list3)
    d= max(list4)
    if a >= b and a >= c and a>=d:
        return list1
    elif b >= a and b >= c and b>=d:
        return list2
    elif c >= a and c >= b and c>=d:
        return list3
    else:
        return list4


#scipt for visualisation of pcb
def visualize_pcb(detections, image_np_with_detections, category_index, detection_threshold):
    viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=500,
                    min_score_thresh=detection_threshold,
                    agnostic_mode=False)
    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()


#script for visualisation of components
def visualize_ocr(img, result):
    #img = cv2.imread(IMAGE_PATH)
    spacer = 100
    for detection in result: 
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        text = detection[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
        img = cv2.putText(img,text,(20,spacer), font, 0.5,(255,0,0),2,cv2.LINE_AA)
        spacer+=15
        
    plt.imshow(img)
    plt.show()