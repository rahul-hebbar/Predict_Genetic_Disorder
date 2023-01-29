import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder

t_df = pd.read_csv("./dataset/pro_test.csv")

patient_col = t_df.pop("Patient Id")

# key = {'Genetic Disorder': array(['Mitochondrial genetic inheritance disorders',
#     	'Multifactorial genetic inheritance disorders',
#        'Single-gene inheritance diseases'], dtype=object), 
# 	'Disorder Subclass': array(["Alzheimer's", 'Cancer', 'Cystic fibrosis', 'Diabetes',
#        'Hemochromatosis', "Leber's hereditary optic neuropathy",
#        'Leigh syndrome', 'Mitochondrial myopathy', 'Tay-Sachs'],
#       dtype=object)}

model = tf.keras.models.load_model("gen_pred.h5")
prediction = model.predict(t_df.values)
print(prediction)