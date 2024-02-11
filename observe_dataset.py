import os
import sys

current_path = os.getcwd()
dataset = os.path.join(current_path, "dataset")
Tuberculosis = os.path.join(dataset, "tuberculosis-phonecamera")


print(len(os.listdir(Tuberculosis))/2)