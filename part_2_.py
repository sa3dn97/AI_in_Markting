import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
import plotly.express as px
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
pd.set_option('display.max_columns', 20)


sale_df = pd.read_csv('sales_data_sample.csv')
# print(sale_df.head())
# print(sale_df.dtypes)
# print(sale_df.isnull().sum())
sale_df['ORDERDATE'] = pd.to_datetime(sale_df['ORDERDATE'])
# print(sale_df.dtypes)
# print(sale_df['ORDERDATE'])
df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
sale_df=sale_df.drop(df_drop,axis=1) # axis = 1 delet all columes
# print(sale_df.isnull().sum())

# print(sale_df.nunique())
# a =[]
# for i in sale_df['COUNTRY']:
#     if i not in a :
#         a.append(i)
# print(a)
# print(len(a))
# print(sale_df['COUNTRY'].value_counts())
# print(sale_df['COUNTRY'].value_counts().index)


def barplot_visualization(x):
    fig = plt.figure(figsize=(12,6))
    fig= px.bar(x=sale_df[x].value_counts().index,y=sale_df[x].value_counts(),color=sale_df[x].value_counts().index,height=600)
    fig .show()

# barplot_visualization('COUNTRY')

# print(sale_df['STATUS'].value_counts())
# print(sale_df['STATUS'].value_counts().index)
sale_df.drop(columns='STATUS',inplace=True)


def dummies(x):
    dummy = pd.get_dummies(sale_df[x])
    sale_df.drop(columns=x,inplace=True)
    return pd.concat([sale_df,dummy],axis=1)

sale_df = dummies('COUNTRY')
sale_df = dummies('DEALSIZE')
sale_df = dummies('PRODUCTLINE')

# print(sale_df.head)

# y= pd.Categorical(sale_df['PRODUCTCODE'])
# print(y)
# y1= pd.Categorical(sale_df['PRODUCTCODE']).codes
# print(y1)

sale_df['PRODUCTCODE'] = pd.Categorical(sale_df['PRODUCTCODE']).codes

sale_df_group = sale_df.groupby(by='ORDERDATE').sum()
# print(sale_df_group)

# fig = px.line(x=sale_df_group.index,y=sale_df_group.SALES ,title='Sales')
# fig.show()

sale_df.drop('ORDERDATE',axis=1,inplace=True)

plt.figure(figsize=(20,20))

corr_metrics = sale_df.iloc[:,:10].corr()
# sns.heatmap(corr_metrics,annot=True,cbar=False,cbar_ax=False)
# plt.show()
sale_df.drop('QTR_ID',axis=1,inplace=True)

import plotly.figure_factory as ff

plt.figure(figsize=(10,10))
for i in range(8):
    if sale_df.columns[i] != ['ORDERLINENUMBER']:
        fig = ff.create_distplot([sale_df[sale_df.columns[i]].apply(lambda x: float(x))],['displot'])
        fig.update_layout(title_text = sale_df.columns[i])
        # fig.show()


plt.figure(figsize=(15,15))
fig = px.scatter_matrix(sale_df,
                        dimensions=sale_df.columns[:8],
                        color='MONTH_ID')

fig.update_layout(
    title = 'Sale_Data',
    width = 1100,
    height = 1100
)

# fig.show()

scaler = StandardScaler()
sale_df_scaled = scaler.fit_transform(sale_df)
print(sale_df_scaled.shape)
#
# scores = []
# range_value  = range(1,15)
# for i in range_value :
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(sale_df_scaled)
#     scores.append(kmeans.inertia_)
#
# plt.plot(scores,'bx-')
# plt.title('Finding right number ')
# plt.xlabel('Clusting ')
# plt.ylabel('scores')
# # plt.show()
#
# kmeans = KMeans(5)
# kmeans.fit(sale_df_scaled)
# labels = kmeans.labels_
# # print(labels)
# # print(kmeans.cluster_centers_.shape)
#
# cluser_center = pd.DataFrame(data=kmeans.cluster_centers_,columns=[sale_df.columns])
# print(cluser_center)
#
# cluser_center = scaler.inverse_transform(cluser_center)
# cluser_center = pd.DataFrame(data=cluser_center,columns=[sale_df.columns ])
# print(cluser_center)
# print(labels.shape)
# # print(labels.max(),labels.min())
#
#
# y_means = kmeans.fit_predict(sale_df_scaled)
# # # print(y_means)
# sale_df_cluster =pd.concat([sale_df,pd.DataFrame({'cluster':labels})],axis=1)
# print(sale_df_cluster )
input_df = Input(shape = (37,))
x = Dense(50, activation = 'relu')(input_df)
x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
encoded = Dense(8, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation = 'relu', kernel_initializer = 'glorot_uniform')(encoded)
x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
decoded = Dense(37, kernel_initializer = 'glorot_uniform')(x)

# autoencoder
autoencoder = Model(input_df, decoded)

# encoder - used for dimensionality reduction
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer = 'adam', loss='mean_squared_error')
# autoencoder.fit(sale_df,sale_df,batch_size=128,epochs=500,verbose=3)
autoencoder.save_weights("autoencoder_1.h5")
pred = encoder.predict(sale_df_scaled)
scores = []

range_values = range(1, 15)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(pred)
  scores.append(kmeans.inertia_)

plt.plot(scores, 'bx-')
plt.title('Finding right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('scores')
plt.show()

kmeans = KMeans(3)
kmeans.fit(pred)
labels = kmeans.labels_
y_kmeans = kmeans.fit_predict(sale_df_scaled)

df_cluster_dr = pd.concat([sale_df, pd.DataFrame({'cluster':labels})], axis = 1)
# print(df_cluster_dr.head())
cluser_centers = pd.DataFrame(data=kmeans.cluster_centers_,columns=[sale_df.columns])
# print(cluser_centers)
cluser_centers = scaler.inverse_transform(cluser_centers)
cluser_centers = pd.DataFrame(data=cluser_centers,columns=[sale_df.columns ])
print(cluser_centers)
