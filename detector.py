import tensorflow as tf
import sys
import time
from observe_dataset import get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset
import cv2
import os
import numpy as np

IMG_HEIGHT = 1224
IMG_WIDTH = 1632

def load_model(model_name):
    start = time.time()
    model = tf.keras.models.load_model(model_name)
    end = time.time()
    print("sucessfully loaded the model")
    print("It takes", end - start, "s to complete")
    return model

def detector_scan_method(model_name, img_path, delta:int):
    detected_coor = []
    model = load_model(model_name)
    width, length, count_bacteria = get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset()
    width = round(width) + delta
    length = round(length) + delta

    img = cv2.imread(img_path)

    x_coor = np.linspace(0, IMG_WIDTH, int(IMG_WIDTH/width))
    y_coor = np.linspace(0, IMG_HEIGHT, int(IMG_HEIGHT/length))
    
    x_coor = np.round(x_coor)
    y_coor = np.round(y_coor)

    for i in range(0, len(y_coor)-1):
        for j in range(0, len(x_coor)-1):
            crop = img[int(y_coor[i]):int(y_coor[i+1]), int(x_coor[j]):int(x_coor[j+1])]

            region_img = cv2.resize(crop, (100,100), interpolation=cv2.INTER_AREA)/255.0

            result = model.predict(region_img.reshape(1, 100, 100, 3))

            if result.argmax() == 1:
                detected_coor.append(int(x_coor[j]))
                detected_coor.append(int(y_coor[i]))
                detected_coor.append(int(x_coor[j+1]))
                detected_coor.append(int(y_coor[i+1]))
    
    print(len(detected_coor)/4)

    i = 0 
    while i < int(len(detected_coor)):
        cv2.rectangle(img, (detected_coor[i], detected_coor[i+1]), (detected_coor[i+2], detected_coor[i+3]), (0,0,255), 2)
        i+=4

    cv2.imwrite("result.jpg", img)
            


def detector_grow_and_shrink_method(model_name, img_path):
    detected_coor = []
    model = load_model(model_name)
    width, length, count_bacteria = get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset()
    width = round(width)
    length = round(length)

    img = cv2.imread(img_path)


def detector_zoom_in_zoom_out_method(model_name, img_path):
    pass

detector_scan_method("model", "dataset/tuberculosis-phonecamera/tuberculosis-phone-0001.jpg", 10)