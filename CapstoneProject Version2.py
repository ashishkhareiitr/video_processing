# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:46:37 2018

@author: Ashish.Khare
"""
from __future__ import print_function
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras import backend as K
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pylab as pl
import matplotlib.pyplot as plt
from statistics import stdev 
from keras.preprocessing.image import img_to_array
#------------------------------------------------------------------------------#

                                #FUNCTION MODULES Start#

# declaration for CNN model
                                
batch_size = 32
num_classes = 2
epochs = 100
data_augmentation = True
num_predictions = 20

############canny edge detection############
 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

##############Key Frames thresold ##########################
def KeyFrameThreshold(VideoPath):

    print('calculating key frame thresold.....')
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)
    cap = cv2.VideoCapture(VideoPath)
    # Read the first frame.
    ret, prev_frame = cap.read()
    prev_edges = auto_canny(prev_frame)
    storediff =[]
    
   
    while ret:
        ret, curr_frame = cap.read()
        
        if ret:
               
            gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            curr_frame_bg = fgbg.apply(blurred)
            #curr_edges = cv2.Canny(curr_frame_bg,100,200)
            curr_edges = auto_canny(curr_frame_bg)
            
            diff = cv2.absdiff(curr_edges, prev_edges)
            non_zero_count = np.count_nonzero(diff)
            storediff.append(non_zero_count)
            #icount = icount +1
            print(non_zero_count)
            prev_edges = curr_edges
                           
            #print(diff)
            #meandiff = meandiff + diff
            #print(meandiff)
            #cv2.imshow('Original',frame) 
      
            # finds edges in the input image image and 
            # marks them in the output map edges 
            #edges = cv2.Canny(frame,100,200) 
      
            # Display edges in a frame 
            #cv2.imshow('Edges',edges) 
            #storediff.append(diff)
    meanstorediff =  np.mean(storediff)
    stdstorediff =   np.std(storediff)
    min_p_frame_thresh = meanstorediff-1.5*stdstorediff
    max_p_frame_thresh = meanstorediff+1.5*stdstorediff 
    
    print('completed calculating key frame thresold.')
    
    return min_p_frame_thresh,max_p_frame_thresh

#Testing
#x,y = KeyFrameThreshold("C:\SAProject\YogaPoses.mp4")
#print(x)
#print(y)


#########################Extracting and saving key frames################################
            
def KeyFrame(VideoPath,KeyFrameFolder,minThreshold,maxThreshold):

    print('extracting key frames.....')
    cap = cv2.VideoCapture(VideoPath)
    # Read the first frame.
    ret, prev_frame = cap.read()
    #prev_frame = cv2.resize(prev_frame, (128, 128))    
    prev_edges = auto_canny(prev_frame)
    
    #prev_edges = cv2.Canny(prev_frame,100,200)
    count = 0
    vfolder = KeyFrameFolder
    os.mkdir(vfolder)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)
    #scaling_factor = 1.5
    while ret:
        ret, curr_frame = cap.read()
        #curr_frame = cv2.resize(curr_frame, (128, 128)) 
        #curr_frame = cv2.resize(curr_frame, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
        if ret:
 
            gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            curr_frame_bg = fgbg.apply(blurred)
            curr_edges = auto_canny(curr_frame_bg)
            
            diff = cv2.absdiff(curr_edges, prev_edges)
            non_zero_count = np.count_nonzero(diff)
            print(non_zero_count)
            
            if  non_zero_count > maxThreshold:
                print('Writing Key-Frame...')
                count = count+1
                cv2.imwrite(os.path.join(vfolder,"frame{:d}.jpg".format(count)), curr_frame) 
                #cv2.imshow('Original',curr_frame) 
            prev_edges = curr_edges    
    print('completed extracting key frames.')    
    return count


###################### clustering dataframes#################

def FrameClustering(FramePath):
 
    train_data = []
    train_labels_one_hot =[]
    trainfileList =[]
    
    for root, dirs, files in os.walk(FramePath):
        for file in files:
            if file.endswith(".jpg"):
                 fullName = os.path.join(root, file)
                 trainfileList.append(fullName)
    
    
    for imagePath in trainfileList:
        
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        image.ndim
        
        image = image.flatten()
        image /= 255
        train_data.append(image)
        #train_data /= 255
        
   
    print(image.shape)
    print(image.dtype)

    dfFrame = pd.DataFrame.from_records(train_data)
    dfFrame.shape

    #Kmeans to label poses  
    kmeans = KMeans(n_clusters=12)  
    kmeansoutput = kmeans.fit(dfFrame) 
    
    clusLabel = kmeans.predict(dfFrame)
    cluCenter = kmeans.cluster_centers_ 
    Lables = pd.DataFrame(clusLabel)
    dfFrame['Cluster'] = clusLabel
    #PCA for visulaisation
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(dfFrame)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2'])
    
    pl.scatter(principalDf['PCA1'],principalDf['PCA2'], c=kmeansoutput.labels_,s=100)

    return dfFrame
    
######################################### train data prep work #####################

def TrainDataPrep(TrainDataPath):

    train_data = []
    train_labels_one_hot =[]
    trainfileList =[]
    
    for root, dirs, files in os.walk(TrainDataPath):
        for file in files:
            if file.endswith(".jpg"):
                 fullName = os.path.join(root, file)
                 trainfileList.append(fullName)
    
    
    for imagePath in trainfileList:
        
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (28,28))
        image = img_to_array(image)
        train_data.append(image)
        path = os.path.split(imagePath)
        x = path[0]
        label = x.split("\\")[4]
        label = int(label)
        train_labels_one_hot.append(label)
        print(imagePath)
        print(label)
        
    train_data = np.array(train_data)
    train_labels_one_hot = np.array(train_labels_one_hot)
    print('train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    
    return train_data,train_labels_one_hot

 
#################################test data prep work ###############################################3

def TestDataPrep(TestDataPath):

    test_data = []
    test_labels_one_hot =[]
    testfileList =[]
    
    for root, dirs, files in os.walk(TestDataPath):
        for file in files:
            if file.endswith(".jpg"):
                 fullName = os.path.join(root, file)
                 testfileList.append(fullName)
    
    
    for imagePath in testfileList:
        
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        #x = x.reshape((1,) + x.shape)
        test_data.append(image)
        test_data[0].shape
        path = os.path.split(imagePath)
        x = path[0]
        label = x.split("\\")[4]
        label = int(label)
        test_labels_one_hot.append(label)
        print(imagePath)
        print(label)
        
    
    test_data = np.array(test_data)
    test_labels_one_hot = np.array(test_labels_one_hot)
    print('test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')  
    
    return test_data,test_labels_one_hot

    
########################CNN model################################################################

def YogaCNNModel(train_data,train_labels_one_hot,test_data,test_labels_one_hot,save_dir,model_name):
    

    
    print('train samples',train_data.shape)
    print('test samples',test_data.shape)
    print('train labels',train_labels_one_hot.shape)
    print('test labels',test_labels_one_hot.shape)
    
    train_labels_one_hot = np_utils.to_categorical(train_labels_one_hot)
    test_labels_one_hot = np_utils.to_categorical(test_labels_one_hot)
    
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(28,28,3)))
     
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    #train_data = train_data.astype('float32')
    #test_data = test_data.astype('float32')
    #train_data /= 255
    #test_data /= 255
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(train_data, train_labels_one_hot,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(test_data, test_labels_one_hot),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            #steps_per_epoch = 1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(train_data)
    
        # Fit the model on the batches generated by datagen.flow(). 
        #steps_per_epoch should be equivalent to the total number of samples divided by the batch size.
        model.fit_generator(datagen.flow(train_data, train_labels_one_hot,
                                         batch_size=batch_size),
                            epochs=epochs,steps_per_epoch=30,
                            validation_data=(test_data, test_labels_one_hot))
    
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    # Score trained model.
    scores = model.evaluate(test_data, test_labels_one_hot, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    #model.save("CBA-model.h5")

    return scores[0],scores[1] 

                                #FUNCTION MODULES end#



#################### run the model####################

### takes 3-4 hrs to complete this.
#def runmodel():
vlist = ["C:\\SAProject\\SuryaNamskar\\yoga1.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga2.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga3.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga4.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga5.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga6.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga7.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga8.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga9.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga10.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga11.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga12.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga13.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga14.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga15.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga16.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga17.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga18.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga19.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga20.mp4",
         "C:\\SAProject\\SuryaNamskar\\yoga21.mp4"
         ]
vlistfolder = ["yoga1","yoga2","yoga3","yoga4","yoga5","yoga6","yoga7","yoga8","yoga9","yoga10","yoga11",
               "yoga12","yoga13","yoga14","yoga15","yoga16","yoga17","yoga18","yoga19","yoga20","yoga21"]

#Testing

for i in range(0,21):
    x,y = KeyFrameThreshold(vlist[i])
    vct = KeyFrame(vlist[i],vlistfolder[i],x,y)

    print('minimum threshold' , x)
    print('maximum threshold',y)
    print('number of key frames extracted',vct)  
    
###############################################################################
    
# clustering frames
trainframe = FrameClustering("C:\\SAProject\\Vid\\train")
testframe = FrameClustering("C:\\SAProject\\Vid\\test")
trainframe.shape[0]
############################


# testing
train_data,train_labels_one_hot = TrainDataPrep("C:\\SAProject\\Vid\\train") 
# testing
test_data,test_labels_one_hot = TestDataPrep("C:\\SAProject\\Vid\\test")  

x,y = YogaCNNModel(train_data,train_labels_one_hot,test_data,test_labels_one_hot,"testCNNModel","testYogaPose.h5")


