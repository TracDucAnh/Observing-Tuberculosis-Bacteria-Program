import os
import sys
import cv2
import numpy as np

def count_imgs():
    current_path = os.getcwd()
    dataset = os.path.join(current_path, "dataset")
    Tuberculosis = os.path.join(dataset, "tuberculosis-phonecamera")
    print (len(os.listdir(Tuberculosis))/2)
    return (len(os.listdir(Tuberculosis))/2)

def get_average_imgs_size():
    length = []
    width = []
    current_path = os.getcwd()
    dataset = os.path.join(current_path, "dataset")
    Tuberculosis = os.path.join(dataset, "tuberculosis-phonecamera")
    for path in os.listdir(Tuberculosis):
        if path[-3:] == "xml":
            pass
        else:
            new_path = os.path.join(Tuberculosis, path)
            img = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
            length.append(img.shape[0])
            width.append(img.shape[1])
    print((sum(length)/len(length), sum(width)/len(width)))
    return (sum(length)/len(length), sum(width)/len(width))
