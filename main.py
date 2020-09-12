from keras.datasets import cifar10
from argparse import ArgumentParser
import random
import cv2 as cv
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.externals import joblib
import sys
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

CLASS_LABELS = ['plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

model_name = 'cifar10_trained_model.h5'
img_width, img_height = 32, 32

def main( args ):

    # Load arguments
    arg_list= sys.argv

    # For training
    if( len(arg_list) == 1) :
        (x_tr, y_tr), (x_tst, y_tst) = cifar10.load_data()
        x_tr = x_tr[:200]
        y_tr = y_tr[:200]
        x_tst = x_tst[:200]
        y_tst = y_tst[:200]
    
    
        y_tr = to_categorical(y_tr, len(CLASS_LABELS))
        y_tst = to_categorical(y_tst, len(CLASS_LABELS))
    
    
        # FEATURE EXTRACTION
        first_convolution = Conv2D(32, (3,3), activation='relu', padding='same',
                                   name='first_convolution', input_shape=(32,32,3))
        second_convolution = Conv2D(32, (3,3), activation='relu', padding='same',
                                   name='second_convolution')
        scaling_down = MaxPooling2D(pool_size=(2,2))
    
        # CLASSIFICATION
        classifier_input = Flatten(name='classifier_input')
        classifier_hidden = Dense(512, activation='relu', name='classifier_hidden')
        overfitting_countermeasure = Dropout(0.5, name='overfitting_countermeasure')
        classifier_output = Dense(len(CLASS_LABELS), activation='softmax', name='classifier_output')
    
        network = Sequential([
            first_convolution,
            second_convolution,
            scaling_down,
            classifier_input,
            classifier_hidden,
            overfitting_countermeasure,
            classifier_output
        ])
        network.summary()
    
        network.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
        network.fit(x_tr, y_tr, batch_size=32, epochs=1, validation_split=0.2)
        network.save(model_name)

    # For testing
    if( len(arg_list) == 2) :
        network = load_model('cifar10_trained_model.h5')
        network.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 
        img = image.load_img('deer.jpg', target_size=(img_width, img_height))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        images = np.vstack([x])
        classes = network.predict_classes(images, batch_size=10)
        print('Image Predicted Class ->>>>---> {}'.format(CLASS_LABELS[classes[0]]))

    if ( len(arg_list) > 2):
        print("INVALID")
    
def parse_arguments():
    parser = ArgumentParser()
    return parser.parse_args()

if __name__ == '__main__':
    main( sys.argv )
