from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten
from keras.models import Model
from keras import optimizers
from keras.layers.core import Flatten, Dense, Dropout
import numpy as np
import scipy
import matplotlib.pyplot as plt
import keras
import keras.regularizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))       
        
class VGGModel:
    def __init__(self):
        self.matrix = []
        model = VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))
#        model.layers.pop()
        output_layer = (Dense(30, activation='relu')(Flatten()(model.output)))
        self.model = Model(inputs=model.input,
                                         outputs=output_layer)       
        for layer in self.model.layers[:-1]:
            layer.trainable = False
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])       
        
    def calibrateBatch(self,batch):
        batch = np.expand_dims(batch,3)
        batch = np.repeat(batch,3,3)
        zeros = np.zeros(batch.shape)
        for i, d in enumerate(batch):
            temp = d.astype('float32')
            zeros[i] =  np.divide(temp,temp.max())
        return preprocess_input(zeros)
    
    def squish(self,img):
        img = img.astype('float32')
        return np.divide(img,img.max())
    
    def calibrateImage(self,img):
        img = self.squish(img)
        img = np.expand_dims(img,2)
        img = np.repeat(img,3,2)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))       
        img = preprocess_input(img)
        return img
        
    def extractFeatures(self,img):      
#        img = self.squish(img)
        if len(img.shape) == 2:       
            img = self.calibrateImage(img)
        else:
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))           
            img = preprocess_input(img)                  
            
        #print(img[0].shape)
        #plt.imshow(img[0])
        #plt.show()
        return self.model.predict(img)
    
    def distance(self,vec1,vec2):
        return scipy.spatial.distance.euclidean(vec1,vec2)
    
    def extractFromBatch(self,batch):
        if len(batch[0].shape) == 2:
            batch = self.calibrateBatch(batch)
        else:
            print("enter here")
            batch = np.array([preprocess_input(i) for i in batch])       
        self.matrix = self.model.predict(batch)
    
    def slice3DScans(self,scans):
        a = 10
        b = int((a*2)/3)-1
        slice_index = round(scans[0].shape[2]/2)              
        return np.asarray(scans[:][:][:][slice_index-a:slice_index+a:b])
    
    def fit(self,scans,labels,label_name,label_value):
        if len(scans[0].shape) == 2:
            X = self.calibrateBatch(scans)
        else:
            X = np.array([preprocess_input(i) for i in scans])
#        for i in X:
#            print(i)
        Y = np.array([[1,0] if i[label_name] == label_value else [0,1] for i in labels])
        
        old_out = self.model.output       
        output_layer = Dense(2, activation='softmax')(old_out)
        self.model = Model(inputs=self.model.input,
                                         outputs=output_layer)
        sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])              
        
        history = LossHistory()

        self.model.fit(X,Y,shuffle=True,batch_size=10,epochs=3,callbacks=[history])
        self.model.layers.pop()
        self.model = Model(inputs=self.model.input,
                                         outputs=self.model.layers[-1].output)
#        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])                     
        return history
        
    def query(self,query_img,k):
        if self.matrix is []:
            print("empty feature matrix")
            return []
        query_img_features = self.extractFeatures(query_img)
        distances_and_indices = []
        for index,database_img_features in enumerate(self.matrix):
            distance = self.distance(query_img_features,database_img_features)
            distances_and_indices.append((distance,index))
        distances_and_indices.sort(key=lambda pair:pair[0])
        relevant_dist_and_indices = distances_and_indices[:k]
        retval_indices = []
        retval_dist = []
        for i in relevant_dist_and_indices:
            dist,index = i
            retval_indices.append(index)
            retval_dist.append(dist)
        return retval_indices,retval_dist
