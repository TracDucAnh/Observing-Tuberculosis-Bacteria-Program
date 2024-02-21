import os
import sys
import cv2
import numpy as np
import math
from random import randint
import time

IMG_HEIGHT = 1224
IMG_WIDTH = 1632

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

def get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset():
    count_bacteria = []
    x_max = []
    x_min = []
    y_max = []
    y_min = []
    corr = []
    current_path = os.getcwd()
    dataset = os.path.join(current_path, "dataset")
    Tuberculosis = os.path.join(dataset, "tuberculosis-phonecamera")
    for path in os.listdir(Tuberculosis):
        if path[-3:] == "xml":
            new_path = os.path.join(Tuberculosis, path)
            xml_file = open(new_path, mode="r")
            coor_path_name = path[0:-4] + ".txt"
            coor_path = os.path.join("Tuberculosis_coordinates", coor_path_name)
            txt_file = open(coor_path, mode = "w")
            header = xml_file.readline()
            count = 0
            negative = False
            while header != "":
                raw_string = header.replace(" ", "")
                if "-" in raw_string:
                    negative = True
                    raw_string = raw_string.replace("-", "")
                if raw_string[:9] in raw_string and raw_string[-8:] in raw_string and raw_string[9:-8].isnumeric():
                    if (negative):
                        corr.append(-int(raw_string[9:-8]))
                        txt_file.writelines("-"+raw_string[9:-8]+"\n")
                        negative = False
                    else:
                        corr.append(int(raw_string[9:-8]))
                        txt_file.writelines(raw_string[9:-8]+"\n")
                if "TBbacillus" in header:
                    count+=1
                header = xml_file.readline()
            count_bacteria.append(count)
            xml_file.close()
            txt_file.close()
    i = 0
    while i < len(corr):
        x_min.append(corr[i])
        y_min.append(corr[i+1])
        x_max.append(corr[i+2])
        y_max.append(corr[i+3])
        i += 4
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    y_min = np.array(y_min)
    y_max = np.array(y_max)
    width = sum(x_max - x_min)/len(x_max - x_min)
    length = sum(y_max - y_min)/len(y_max - y_min)
    print("successfully extract bacteria's coordinates")
    return width, length, count_bacteria
    
def crop_bacteria_images():
    # crop bactreia images
    print("Start croping images of bacteria...")
    print("Croping bacteria from images")
    start = time.time()
    current = os.getcwd()
    coor_folder = os.path.join(current, "Tuberculosis_coordinates")
    count = 0
    save_index = 1
    for txt in os.listdir(coor_folder):
        coor = []
        img_name = txt[:-4]+".jpg"
        img_path = os.path.join("dataset", "tuberculosis-phonecamera", img_name)
        img = cv2.imread(img_path)
        txt_path = os.path.join(coor_folder, txt)
        txt_file = open(txt_path, mode="r")
        header = txt_file.readline()
        while header != "":
            coor.append(int(header))
            header = txt_file.readline()
        i = 0
        if len(coor) != 0:
            while i < len(coor):
                crop = img[coor[i+1]:coor[i+3], coor[i]:coor[i+2]]
                i += 4
                try:
                    crop = cv2.resize(crop, (100,100), interpolation=cv2.INTER_AREA)
                except:
                    pass
                count += 1
                save_path = os.path.join("tuberculosis_set", "tuberculosis_"+str(count)+".jpg")
                try:
                    cv2.imwrite(save_path, crop)
                except:
                    pass
    end = time.time()
    print(f"successfully crop {count} files of bacteria samples")
    print("It takes", end - start, "s to complete")

def crop_environment_images(percentage):
    print("Start croping images of environments...")
    start = time.time()
    current = os.getcwd()
    img_folder = os.path.join(current, "removed_bacteria_out_from_images")
    index = 0
    for img_file in os.listdir(img_folder):
        img_file = os.path.join(img_folder, img_file)
        img = cv2.imread(img_file)
        x = np.round(np.linspace(0, IMG_WIDTH, int(IMG_WIDTH/100)))
        y = np.round(np.linspace(0, IMG_HEIGHT, int(IMG_HEIGHT/100)))
        grid = len(x)*len(y)
        crop_number = int(grid*percentage)
        count = 0
        for i in range(len(y)-1):
            if count < crop_number:
                for j in range(len(x)-1):
                    if np.zeros(3, dtype=int) in img[int(y[i]):int(y[i+1]), int(x[j]):int(x[j+1])]:
                        pass
                    else:
                        crop = img[int(y[i]):int(y[i+1]), int(x[j]):int(x[j+1])]
                        name = "none_"+str(index)+".jpg"
                        save_path = os.path.join("environment_set", name)
                        cv2.imwrite(save_path, crop)
                        count += 1
                        index += 1
            else:
                break
    end = time.time()
    print(f"successfully crop {index} files of environment samples with {percentage*100}% per images")
    print(f"It takes {end - start}s to complete")