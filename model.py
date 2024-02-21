import numpy as np
import cv2
import os
import sys
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import time

IMG_HEIGHT = 100
IMG_WIDTH = 100
EPOCHS = 15
TEST_SIZE = 0.3
RANDOM_STATE = 64

def main():
    if len(sys.argv) not in [2, 3, 4]:
        sys.exit("Usage: python model.py environment_set tuberculosis_set [model.h5]")

    start = time.time()
    img, label = load_data(sys.argv[1], sys.argv[2], 50000)

    img = np.array(img)
    label = np.array(label)

    print("Img shape: ", img.shape)
    print("Label shape: ", label.shape)

    x_train, x_test, y_train, y_test = train_test_split(img, label, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    x_train = x_train/255.0
    x_test = x_test/255.0
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test =  tf.keras.utils.to_categorical(y_test)

    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)

    end = time.time()

    print("It takes", end-start, "s to prepare the data for training")

    start = time.time()

    model = get_model()

    model.fit(x_train, y_train, epochs = EPOCHS)

    model.evaluate(x_test, y_test, verbose = 2)

    end = time.time()

    print("It takes", end-start, "s to train the model for", EPOCHS, "epochs")

    if len(sys.argv) == 4:
        filename = sys.argv[3]
        model.save(filename)
        print(f"Model saved to {filename}")

def load_data(environment, bacteria, limit_environment):
    start = time.time()
    print("Loading data...")
    current = os.getcwd()
    try:
        env_folder = os.path.join(current, environment)
    except:
        sys.exit("Enviroment set does not exist")
    
    try:
        bac_folder = os.path.join(current, bacteria)
    except:
        sys.exit("Bacteria set does not exist")

    img = []
    label = []
    count = 0
    for env_file in os.listdir(env_folder):
        env_file = os.path.join(env_folder, env_file)
        image = cv2.imread(env_file)
        image = cv2.resize(image, (100,100), interpolation=cv2.INTER_AREA)
        img.append(image)
        label.append(0)
        count+= 1
        print(f"Environment {count}")
        if count > limit_environment:
            break
    print("successfully loaded environment samples")
    count = 0
    for bac_file in os.listdir(bac_folder):
        bac_file = os.path.join(bac_folder, bac_file)
        image = cv2.imread(bac_file)
        img.append(image)
        label.append(1)
        count+=1
        print(f"bacteria {count}")
    print("successfully loaded bacteria samples")
    end = time.time()
    print(f"It takes {end - start} to complete")
    return img, label
    
def get_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        70, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(
        50, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    tf.keras.layers.Conv2D(
        30, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),


    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(100, activation="relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(300, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(300, activation = "relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.summary()

    return model

if __name__ == "__main__":
    main()