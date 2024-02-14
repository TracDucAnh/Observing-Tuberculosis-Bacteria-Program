import os
import sys
import cv2
import numpy as np
import math

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
            coor_path = os.path.join("Tuberculosis_coordinate", coor_path_name)
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
    
