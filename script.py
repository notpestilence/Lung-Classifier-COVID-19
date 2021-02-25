#!/usr/bin/env python
# coding: utf-8

# # Lung Condition Classification on COVID-19 Image Dataset
# ---
# # 1. Introduction
# 
# This computer vision-based project is focused on model building towards multi-class classification in identifying whether a patient has one of three lung conditions:
# 1. Infected with COVID-19
# 2. Viral Pneumonia
# 3. Normal
# 
# Done with convolutional neural networks (CNN), the goal of the model is to provide a diagnosis based on a patient’s X-ray scan, with high-AUC and classification accuracy.
# 
# Hopefully, the model I created will benefit both medical staff as well as data enthusiasts like myself, as the challenge of interpreting X-ray scans are often steep without much domain expertise in pulmonology and/or virology. This is where the neural network comes in. 
# 
# > **NOTE: Please do not test diagnostic performance of a model without an extensive clinical study!**
# 
# *At any rate, please consider this to be my modest form of contribution towards addressing the COVID-19 pandemic that's currently ravaging the world.*
# 
# ## 1.1. Source & Acknowledgements
# 
# The dataset was uploaded to [Kaggle](https://www.kaggle.com/pranavraikokte/covid19-image-dataset) by the user [Pranav Raikote](https://www.kaggle.com/pranavraikokte) around April 2020, but only gained engagement in later months up until today (25 Feb 2021). 
# 
# Original data is sourced from University of Montreal via their publicly accessible [GitHub repository](https://github.com/ieee8023/covid-chestxray-dataset).
# 
# ## 1.2. Data License
# 
# This data is licensed under **Creative Commons Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0)**
# 
# ![license.png](attachment:license.png)
# 
# ## 1.3. Framework Summary
# 
# The model will be built around `Keras`, specifically wrapped with stacked `Conv2D()` and `MaxPooling2D()` layers before eventually passing the flattened feature matrices onto `Dense()` layers for final classification. 
# 
# For labeling, I'll be using Sparse Categorical Cross-Entropy, which uses a single integer for a class label, rather than a whole vector (and skipping one-hot encoding) so as to not take up computation demand from my local machine.
# 
# ---
# 
# # 2. Image Data Loading
# 
# First, we import the essential modules:

# In[57]:


import os, sys

import pandas as pd
import numpy as np
import tensorflow as tf


# A brief preview of the images this project will be working around:
# 
# ![samples.jpg](attachment:samples.jpg)

# Load the images onto `tensorflow` before pre-processing. We split training set into 80% training and 20% validation.

# In[56]:


train_loader = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = (1.0 / 255), # Pixel normalization
    validation_split = 0.2, # Splits into 20% validation set
    zoom_range=0.2, # Augmentation parameter
    rotation_range=15, # Augmentation parameter
    width_shift_range=0.05, # Augmentation parameter
    height_shift_range=0.05) # Augmentation parameter

test_loader = tf.keras.preprocessing.image.ImageDataGenerator(rescale = (1.0 / 255))

print("For training set: ")
train_iterator = train_loader.flow_from_directory('data/train',
                                                     class_mode='categorical',
                                                     color_mode='grayscale',
                                                     target_size = (512, 512),
                                                     subset = "training",
                                                     # save_to_dir = 'data/debug' to manually check transformed images
                                                     batch_size=16)

print("\nFor validation set: ")
val_iterator = train_loader.flow_from_directory('data/train',
                                                     class_mode='categorical',
                                                     color_mode='grayscale',
                                                     target_size = (512, 512),
                                                     subset = "validation",
                                                     # save_to_dir = 'data/debug' to manually check transformed images
                                                     batch_size=16)
print("\nFor test set: ")
test_iterator = test_loader.flow_from_directory("data/test",
                                                class_mode='categorical',
                                                target_size = (512, 512),
                                                color_mode='grayscale', batch_size = 66)

x_test, y_test = test_iterator.next()


# In[4]:


# Quick sanity check
print("Dimensions of one sample in the training set:")
print(train_iterator.next()[0].shape)

print("\nDimensions of one label in the training set:")
print(train_iterator.next()[1].shape)

print("\nCheck respective class labels:")
print(list(train_iterator.class_indices.items()))


# The tensors in the test set represents the light intensity (as channels) in the images. As x-ray scans are considered *grayscale* in nature, we only have one channel per image. 
# 
# * As shown in the `print` statement below, the closer the number is to `1`, the closer it is to *white* (and vice versa).
# * In the test label preview, this particular datapoint belongs to class `1`, indicating this x-ray scan belongs to a person without COVID-19 nor Viral Pneumonia.

# In[30]:


print("Preview of test set: {}".format(x_test[0][0][40:50])) # This represents the light intensity in the images.
print("Preview of test labels: {}".format(y_test[0])) # This represents the label.


# ---
# ## 2.1. Tensor Shape Definition
# For sample batch input:
# * `16` is the number of images per batch;
# * `512 x 512` is the dimensions of the images, rescaled, overriding previous aspect ratio;
# * `1` is the number of channels; because this is grayscale, 1 represents only one channel, the light channel. 
# 
# For sample label:
# * `16` is the number of images per batch;
# * `3` is the number of classes:
# 
# 
# > * Class `0` refers to `Covid`
# > * Class `1` refers to `Normal`
# > * Class `2` refers to `Viral Pneumonia`
# 
# ## 2.2. Model Building
# 
# Stacking `Conv2D()` and `MaxPooling2D()` layers simultaneously are the main idea behind this whole network. Also, with `strides` set to `1` on convolutional layers, the model prevents any loss of information at the cost of training time.
# The full structure of the model can be seen below:
# 
# ![model.png](attachment:model.png)

# In[6]:


model = tf.keras.Sequential(name = "LungClassifier")
model.add(tf.keras.Input(
    shape=(512, 512, 1), name = "InputLayer"))
model.add(tf.keras.layers.Conv2D(
    4, 5, strides=1, activation="relu", name = "1stConvLayer")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(3, 3), strides=(3, 3), name = "1stPoolLayer"))
model.add(tf.keras.layers.Conv2D(
    4, 3, strides=1, activation="relu", name = "2ndConvLayer")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2), name = "2ndPoolLayer"))
model.add(tf.keras.layers.Flatten(name = "Flatten"))

## Wrap up the model building process before compiling

model.add(tf.keras.layers.Dense(3, activation="softmax", name = "FinalClassifier"))

(model.summary())


# Even with this simple model, the network has a whopping amount of parameters to train with 82,923 total trainable weights. In this particular project, the input size is around `512 x 512` pixels which is a lot of tensors to compute per epoch. This is why I don't add any more deeper layers after the second pooling layer `2ndPoolLayer`, or even add another `Dense` before the final classifier `FinalClassifier`. 
# 
# If I had done so, the model would possess an exponentially large number of weights to train, around 100-300 thousand.
# 
# ---
# 
# ## 2.3. Model Training
# 
# Some final configurations needed to be made before starting to fit our image tensors into the instantiated model:
# * Define `EarlyStopping()` callback as a method to prevent overfitting towards the training dataset. Additional parameters:
#    - Monitoring validation area under ROC curve, `val_auc`;
#    - Letting the callback run for `10` more passes to see whether `val_auc` experiences a significant increase.
#    
#   However, this callback eventually was not used in the final model training and was only used to monitor performance before we iterate too much and yielded too little.
#   
#   
# * Define `ModelCheckpoint()` callback for saving learned kernels/filters along the epochs. Additional parameters:
#   - Monitoring validation multi-categorical accuracy, `val_categorical_accuracy`;
#   - Saving only the model with most categorical accuracy along the epochs.
#   
#   
# * Setting `Adam` as the network optimizer with learning rate configured to `0.005`.
# * Setting categorical cross-entropy as the loss function.

# In[ ]:


## Run this cell to start training
# Define earlystopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=10)

# Define model checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
  "model_checkpoints", monitor="val_categorical_accuracy", verbose=1, save_best_only=True, mode="max")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005), # Tweak learning rate as needed
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = [
      tf.keras.metrics.CategoricalAccuracy(),
      tf.keras.metrics.AUC() 
      ]
)

model.fit(
        train_iterator,
        steps_per_epoch= train_iterator.samples/16, # Samples divided by batch size
        epochs = 40,
        validation_data = val_iterator,
        validation_steps = val_iterator.samples/16, # Samples divided by batch size
        callbacks = [checkpoint])


# *Note: output of the training cell is collapsed to save space.*
# 
# ---
# ## 2.4. Model Evaluation
# With this simple model, the evaluation metrics for validation dataset comes out surprisingly good with:
# * ~0.94 area under ROC curve (perfect AUC is 1)
# * ~90% multi-categorical prediction accuracy (perfect accuracy is 100%)
# 
# As a side note, a baseline model would only possess 33.3% multi-categorical prediction accuracy.
# 
# After the model performance is deemed reasonable, we evaluate on the test set of total 60+ images, as well as calculate ground truth versus prediction labels with `sklearn`'s `classification_report`.

# In[22]:


cross_entropy, acc, auc = model.evaluate(x_test, y_test)
print("------------------ EVALUATION FINISHED! ------------------".center(115))
print("""Final Multi-Categorical Cross-Entropy (loss func.) is {}
Final Multi-Categorical Accuracy (eval. metric) is {}
Final Area Under ROC Curve (eval. metric) is {}""".format(cross_entropy, acc, auc))


# Calculating precision, recall, and F1-score of the three classes. As we can see, this dataset is totally not biased as there are rougly the same amount of images in each class. 
# 
# The most important result is the F1-score, which uses the harmonic mean of all predictions in all classes. F1-score represents the overall *viability* of the model.

# In[25]:


from sklearn.metrics import classification_report
y_estimate = model.predict(x_test)
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(y_test, axis = 1)
print(classification_report(y_true, y_estimate))


# ---
# # 3. Conclusion
# 
# ## 3.1. Model Performance
# With only a simple architecture, this CNN model reached the following metrics tested on unseen data:
# * **84% classification accuracy**, this means the trained model predicts the correct class 84% of the time.
# * **0.94 AUC of the ROC curve.**, this means for a random, unseen x-ray, there is a 94% chance the model would predict towards a true class than to a false one.
# 
# ## 3.2. Recommendation
# For future modelling, I would absolutely advise to normalize all the input tensors first (scaled to floats 0-1); as neural networks struggle with large integers in computing their weights/kernels. Doing this, whilst may seem trivial at first, can potentially increase evaluation metrics up to 10-20% on validation data set.
# 
# The pre-trained model had also been exported, in case anyone in the data science community would like to train it further with larger datasets.

# In[ ]:


# Run this cell to export model

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=216,
)


# <p style="text-align: center;"><b><i>--- Written, tested, and published by Charis Chrisna ---</b></i> <br> <a href="http://notpestilence.github.io">(Portfolio)</a></p>
