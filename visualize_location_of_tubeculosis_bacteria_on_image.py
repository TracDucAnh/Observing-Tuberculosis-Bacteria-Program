import os
import cv2
from observe_dataset import get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset, crop_bacteria_images, crop_environment_images
import time
import numpy as np
start = time.time()
width, length, count_bacteria = get_average_size_of_Tuberculosis_and_extract_corrdinate_from_dataset()
end = time.time()
print("It takes", end-start, "s to complete")

start = time.time()
txt_folder = os.path.join("Tuberculosis_coordinates")
for txt in os.listdir(txt_folder):
    tuberculosis_coor = []
    txt_file_path = os.path.join(txt_folder, txt)
    txt_file = open(txt_file_path, mode = "r")
    line = txt_file.readline()
    while line != "":
        tuberculosis_coor.append(int(line))
        line = txt_file.readline()
    
    img_name = txt[0:-4]+".jpg"

    img_path = os.path.join("dataset", "tuberculosis-phonecamera", img_name)

    image = cv2.imread(img_path)

    img_copy = cv2.imread(img_path)

    i = 0

    while i < len(tuberculosis_coor):
        cv2.rectangle(image, (tuberculosis_coor[i], tuberculosis_coor[i+1]), (tuberculosis_coor[i+2], tuberculosis_coor[i+3]), (0,255,0), 2)
        black_shape = img_copy[tuberculosis_coor[i+1]:tuberculosis_coor[i+3], tuberculosis_coor[i]:tuberculosis_coor[i+2]].shape
        img_copy[tuberculosis_coor[i+1]:tuberculosis_coor[i+3], tuberculosis_coor[i]:tuberculosis_coor[i+2]] = np.zeros(black_shape, dtype=int)
        i += 4

    
    new_path = os.path.join("sample", txt[0:-4]+"_"+str(len(tuberculosis_coor)/4)+".jpg")
    new_black_path = os.path.join("removed_bacteria_out_from_images", txt[0:-4]+"_"+str(len(tuberculosis_coor)/4)+".jpg")
    cv2.imwrite(new_black_path, img_copy)
    cv2.imwrite(new_path, image)
end = time.time()
print("successfully locate tuberculosis bacteria on dataset's images")
print("It takes", end-start, "s to complete")

start = time.time()
crop_environment_images()
end = time.time()
print("It takes", end-start, "s to complete")