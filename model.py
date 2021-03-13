import numpy as np
import keras
import csv
import os
import matplotlib.pyplot as plt
import glob

from scipy import ndimage

###INFO###
'''
The data used for perform training to the model is located in:
/opt/carnd_p3/

It is possible to create a Model from Scratch or load one.
ldmodel = False ##If you want to create the model from Scratch
ldmodel = True ##If you want to load a previously created (and trained) one.
'''
##########

ldmodel = True ##Select if you want to load a model
model_name = 'model.h5' ##Name of the model you want to load

folder = 'Curves1' #Write the name of the folder of the data for training
train_file = '/opt/carnd_p3/' + folder + '/driving_log.csv' ##DON´T Change


###Read the CSV and separate in files
lines = []
with open(train_file) as csvfile:
     reader = csv.reader(csvfile)
     for line in reader:
         lines.append(line)
            
del lines[0] #Delete the first element of the list, is the element that contains the titles
    
###Get the images and labels for the training.    
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/opt/carnd_p3/' + folder + '/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

##Imports for creating a model
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Loads a model in case you previouly selected that.
if ldmodel == True:
    model = load_model(model_name)
#Creates a model (NVIDIA architecture)
else:
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #Normalize Data
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(Dropout(0.2))
    model.add(Dense(50))
    #model.add(Dropout(0.2))
    model.add(Dense(10))
    #model.add(Dropout(0.2))
    model.add(Dense(1))

##Perform the training
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

#Save the Model
model.save('model.h5')

'''
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''

