import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("./dataset/pro_train.csv")

enc = LabelEncoder()
out_key = {}
for i in ['Genetic Disorder', 'Disorder Subclass']:
	enc.fit(df[i].values)
	df[i] = enc.transform(df[i].values)
	out_key[i] = enc.classes_

print(out_key)
# Splitting Training and Testing Data
data_size = df.shape[0]
train_size = int(0.95 * data_size)
train_inp_df = df.iloc[:train_size,:-2]
train_tar_df = df.iloc[:train_size,-2:]
test_inp_df = df.iloc[train_size:,:-2]
test_tar_df = df.iloc[train_size:,-2:]
print(train_inp_df.shape,train_tar_df.shape)
print(test_inp_df.shape,test_tar_df.shape)

# Model
model = tf.keras.Sequential([
  layers.Dense(30,input_dim=30,kernel_initializer='normal',activation='relu'),
  layers.Dense(2048, activation='relu'),
  layers.Dense(2)
])

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

print(model.summary())
model.fit(x=train_inp_df.values,y=train_tar_df.values,validation_split=0.2,epochs=30,batch_size=150)

loss, accuracy = model.evaluate(x=test_inp_df,y=test_tar_df)
print("Accuracy", accuracy)

model.save('gen_pred.h5')