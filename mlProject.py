
import numpy as np
import pandas as pd
import os
import cv2
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import random
from tensorflow import keras
from tensorflow.keras import layers






df = pd.read_csv(r'C:\Image\malicious_phish.csv', nrows=10000)


voc = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","_","0","1","2","3","4","5","6","7","8","9","-",";",".","!","?",":",",","\\","\"","/","|","","@","#","$","%","^","&","*","~","`","+","=","<",">","(",")","[","]","{","}"," "]
voc64 = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","_","0","1","2","3","4","5","6","7","8","9","-",";",".","!","?",":",",","\\","\"","/","","@","#","$","%","&","*","~","+","=","<",">","(",")","[","]"]
voc = voc64
voclen = len(voc)
maxlen= 24
char_to_int = dict((c, i) for i, c in enumerate(voc))

abs_path = os.path.dirname(os.path.abspath(__file__))
path_save = os.path.join(abs_path,"convImage")

class_to_int = {'benign': 0, 'phishing': 1, 'defacement': 2, 'malware': 3}  # Replaced class names as integers 

labels = []
opis = []
prefix = 'training'
for index, row in df.iterrows():
    tocode = row['text']
    tocode = tocode.replace('˙','')
    integer_encoded = [char_to_int[char] for char in tocode if char in voc]

    image = np.zeros((voclen,maxlen,1), np.uint8)

    for z in range(len(integer_encoded)):
        if(z > maxlen-1):
            break
        image[integer_encoded[z], z, 0]=255
    
    label = row['label'] # Label is the name of the column includes classes 
    label_int = class_to_int[label]  # Map class name to integer
    ntem = f'{label_int}_{index}{prefix}.png'
    cv2.imwrite(os.path.join(path_save,'image',ntem), image)
    opis.append(f'...\\image\\{ntem}\t{label_int}\n')
    labels.append(label_int)


file = open(os.path.join(path_save,prefix + ".map"), 'w', encoding='utf-8')
file.writelines(opis)
file.close()


# Set the path to the directory containing the PNG files
path_to_images = "C:\Image\mlProject\convImage\image"

# Define the function to load the dataset
def load_dataset():
    images = []
    labels = []
    
    # Load images and labels
    for filename in os.listdir(path_to_images):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(path_to_images, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (24, 63))  # Resize images to a standard size
            images.append(img)
            labels.append(int(filename.split("_")[0]))
            
    return images, labels

# Call the load_dataset function to get the images and labels
images, labels = load_dataset()

# Split the dataset into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model / data parameters
num_classes = 4
input_shape = (63, 24, 1)




# Scale images to the [0, 1] range
train_images = np.array(train_images, dtype="float32") / 255
test_images = np.array(test_images, dtype="float32") / 255
# Make sure images have shape (28, 28, 1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)
print("train_images shape:", train_images.shape)
print(train_images.shape[0], "train samples")
print(test_images.shape[0], "test samples")


# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)



model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()



batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# Train the model
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)



score = model.evaluate(test_images, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


 # save model
model.save('MLfinal_model.h5')
 



# Load the trained model
model = keras.models.load_model('C:\Image\mlProject\MLfinal_model.h5')




# Encode the input URL into an image using the same encoding scheme
input_text = input("Enter url: ")

char_to_int = dict((c, i) for i, c in enumerate(voc))
integer_encoded = [char_to_int[char] for char in input_text if char in voc]
image = np.zeros((voclen,maxlen,1), np.uint8)
for z in range(len(integer_encoded)):
    if(z > maxlen-1):
        break
    image[integer_encoded[z], z, 0]=255
image = cv2.resize(image, (24, 63))
image = np.expand_dims(image, -1)
image = np.expand_dims(image, 0)

# Define class label mapping
class_labels = {
    0: 'benign',
    1: 'phishing',
    2: 'defacement',
    3: 'malicious'
}

# Make the prediction
predictions = model.predict(image)
predicted_class = np.argmax(predictions[0])

# Look up the corresponding class label from the dictionary
predicted_label = class_labels[predicted_class]

# Generate the output message
output_message = f"The URL you provided leads you into a {predicted_label} website."
print(output_message)

