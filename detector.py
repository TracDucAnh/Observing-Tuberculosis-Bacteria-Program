import tensorflow as tf
import sys
import time
from observe_dataset import get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset
import cv2
import os
import numpy as np
import time

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

    cv2.imwrite(os.path.join("result", "result.jpg"), img)
            


def detector_grow_and_shrink_method(model_name, img_path, flex:int, threshold:int):
    start = time.time()
    detected_coor = []
    model = load_model(model_name)
    # width, length, count_bacteria = get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset()
    # width = round(width)
    # length = round(length)

    img = cv2.imread(img_path)

    flex_width = flex
    flex_length = flex

    x_coor = np.linspace(0, IMG_WIDTH, int(IMG_WIDTH/flex_width))
    y_coor = np.linspace(0,IMG_HEIGHT, int(IMG_HEIGHT/flex_length))

    x_coor = np.round(x_coor)
    y_coor = np.round(y_coor)

    # flex vertically
    for i in range(0, len(y_coor)-1):
        mid_y = int((y_coor[i+1]+y_coor[i])/2)
        for j in range(0,len(x_coor)-1):
            mark_not_found = False
            mid_x = int((x_coor[j+1]+x_coor[j])/2)
            for v in range(0, int(flex_width/threshold)):
                if flex_width <= threshold:
                    mark_not_found = True
                    break
                else:
                    x_coor_new_start = mid_x - int(flex_width/2)
                    x_coor_new_end = mid_x + int(flex_width/2)
                    y_coor_new_start = mid_y - int(flex_length/2)
                    y_coor_new_end = mid_y + int(flex_length/2)
                    if y_coor_new_start < 0:
                        y_coor_new_start = 0
                    if y_coor_new_end > IMG_HEIGHT:
                        y_coor_new_end = IMG_HEIGHT
                    if x_coor_new_start < 0:
                        x_coor_new_start = 0
                    if x_coor_new_end > IMG_WIDTH:
                        x_coor_new_end = IMG_WIDTH
                    crop = img[y_coor_new_start:y_coor_new_end, x_coor_new_start:x_coor_new_end]
                    crop = cv2.resize(crop, (100,100), interpolation=cv2.INTER_AREA)/255.0
                    result = model.predict(crop.reshape(1, 100, 100, 3))
                    if result.argmax() == 1:
                        detected_coor.append(x_coor_new_start)
                        detected_coor.append(y_coor_new_start)
                        detected_coor.append(x_coor_new_end)
                        detected_coor.append(y_coor_new_end)
                        mark_not_found = False
                        break
                    flex_width -= threshold
                    flex_length += threshold
            flex_width = flex
            flex_length = flex
            for v in range(int(flex_length/threshold)):
                if mark_not_found == False:
                    break
                else:
                    if flex_length <= threshold:
                        mark_not_found =False
                        break
                    else:
                        x_coor_new_start = mid_x - int(flex_width/2)
                        x_coor_new_end = mid_x + int(flex_width/2)
                        y_coor_new_start = mid_y - int(flex_length/2)
                        y_coor_new_end = mid_y + int(flex_length/2)
                        if y_coor_new_start < 0:
                            y_coor_new_start = 0
                        if y_coor_new_end > IMG_HEIGHT:
                            y_coor_new_end = IMG_HEIGHT
                        if x_coor_new_start < 0:
                            x_coor_new_start = 0
                        if x_coor_new_end > IMG_WIDTH:
                            x_coor_new_end = IMG_WIDTH
                        crop = img[y_coor_new_start:y_coor_new_end, x_coor_new_start:x_coor_new_end]
                        crop = cv2.resize(crop, (100,100), interpolation=cv2.INTER_AREA)/255.0
                        result = model.predict(crop.reshape(1, 100, 100, 3))
                        if result.argmax() == 1:
                            detected_coor.append(x_coor_new_start)
                            detected_coor.append(y_coor_new_start)
                            detected_coor.append(x_coor_new_end)
                            detected_coor.append(y_coor_new_end)
                            mark_not_found = False
                            break
                        flex_width+=threshold
                        flex_length-=threshold
    print(len(detected_coor)/4)

    i = 0 
    while i < int(len(detected_coor)):
        cv2.rectangle(img, (detected_coor[i], detected_coor[i+1]), (detected_coor[i+2], detected_coor[i+3]), (0,0,255), 2)
        i+=4

    cv2.imwrite("result.jpg", img)
    end = time.time()
    print("It takes", end-start, "s to complete")



def detector_zoom_in_zoom_out_method(model_name, img_path):
    # TODO
    pass