#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from sklearn. impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn.metrics as metrics


# # 1. Data Loading

# In[2]:


dataset = pd.read_csv('datasets/data.csv')
# link dataset: https://www.kaggle.com/datasets/shree1992/housedata?datasetId=46927
dataset


# # 2. Data Understanding

# In[3]:


dataset.info()


# ### Deskripsi variabel:
# - date: tanggal dibuatnya data
# - price: harga rumah dalam dolar Amerika Serikat (USD)
# - bedrooms: jumlah kamar tidur 
# - bathrooms: junlah kamar mandi
# - sqft_living: luas tempat tinggal / rumah dalam satuan kaki persegi
# - sqft_lot: luas tanah dalam satuan kaki persegi
# - floors: ukuran lantai 
# - waterfront: varibel kategorik apakah rumah dekat tepi laut atau tidak
# - view: varibel kategorik apakah rumah memiliki view bagus atau tidak
# - condition: varibel kategorik tentang kondisi rumah, semakin besar angka maka semakin bagus kondisinya
# - sqft_above: luas atap dalam satuan kaki persegi
# - sqft_basement: luas basement dalam satuan kaki persegi
# - yr_built: tahun pembangunan rumah
# - yr_renovate: tahun renovasi rumah
# - street: alamat rumah
# - city: kota lokasi rumah
# - statezip: kodepos
# - country: negara

# In[4]:


dataset.describe()


# bisa dilihat pada hasil data describe diatas didapat informasi:
# - terdapat kolom yang tidak tercantum dalam tabel diatas merupakan kolom yang tidak relevan terhadap dataset seperti kolom date dan country sehingga akan dihapus.
# - kolom street dan city sudah diwakilkan dengan statezip sehingga akan dihapus.
# - terdapat nilai 0 atau null pada kolom price. Untuk mengatasinya, akan kita lakukan setelah ini.
# - kolom statezip tidak terlihat dalam informasi diatas karena merupakan variabel kategorik. Untuk mengatasinya akan dilakukan proses onehotencoding nanti.

# ## Hapus kolom yang tidak relevan
# Penghapusan kolom ini dilakukan karena kolom - kolom tersebut nantinya tidak akan digunakan dalam permodelan ML sehingga pilihan satu - satunya hanya menghapus kolom tersebut.

# In[5]:


dataset.drop(['date', 'street', 'city', 'country'], axis = 1, inplace = True)
dataset


# ## Mendeteksi data duplikasi dan missing value

# In[6]:


duplicate_rows_dataset = dataset[dataset.duplicated()]
print("number of duplicate rows: ", duplicate_rows_dataset.shape)


# In[7]:


print(dataset.isnull().sum())


# Didapat informasi bahwa dataset tidak memiliki missing value ataupun duplikasi.

# ## Mengatasi nilai outlier pada data

# In[8]:


sns.boxplot(x=dataset['sqft_living'])


# In[9]:


sns.boxplot(x=dataset['sqft_lot'])


# In[10]:


sns.boxplot(x=dataset['sqft_above'])


# In[11]:


sns.boxplot(x=dataset['sqft_basement'])


# Didapat informasi bahwa terdapat nilai outliers pada dataset. Seltman dalam “Experimental Design and Analysis” menyatakan bahwa outliers yang diidentifikasi oleh boxplot (disebut juga “boxplot outliers”) didefinisikan sebagai data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1. Jadi kita akan Kita akan menggunakan implementasi ini untuk menghilangkan nilai outlier pada baris yang memiliki nilai dibawah 1.5Q1 atau diatas 1.5Q3. 

# In[12]:


Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR=Q3-Q1
dataset=dataset[~((dataset<(Q1-1.5*IQR))|(dataset>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah kita drop outliers
dataset.shape


# ## Univariate Analysis

# In[13]:


dataset.hist(bins=50, figsize=(20,15))
plt.show()


# ## Analisis Multivariate

# In[14]:


# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(dataset, diag_kind = 'kde')


# In[15]:


plt.figure(figsize=(10,5))
c= dataset.corr().round(2)
sns.heatmap(c, cmap='coolwarm', linewidths=0.5, annot=True)
plt.title("Correlation Matrix untuk Fitur", size=20)


# Berdasarkan hasil analisis univariate dan multivariate, didapatkan informasi sebagai berikut:
# - fitur kategorik waterfront dan view tidak relevan terhadap dataset sehingga bisa dihapus
# - fitur yang memiliki pengaruh paling besar terhadap harga yaitu fitur sqft_living(luas rumah satuan kaki persegi)

# In[16]:


dataset.drop(['waterfront', 'view'], axis = 1, inplace = True)
dataset


# # 3. Data Preparation

# ## One Hot Encoding

# In[17]:


dataset.statezip.value_counts()


# In[18]:


le = LabelEncoder()
dataset['statezip_encoded'] = le.fit_transform(dataset.statezip)
dataset.head()


# In[19]:


dataset.statezip_encoded.value_counts()


# In[20]:


dataset.drop(['statezip'], axis = 1, inplace = True)
dataset.head()


# ## Train Test Split 

# In[21]:


X = dataset.drop(["price"],axis =1)
y = dataset["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# In[22]:


print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# ## StandardScaler

# In[23]:


numerical_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'statezip_encoded']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()


# In[24]:


X_train[numerical_features].describe().round(4)


# # 4. Model Development

# In[25]:


# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['LinearRegression', 'kNN', 'RandomForest'])
models


# ## Linear Regression

# In[26]:


LR = LinearRegression()
LR.fit(X_train, y_train)

models.loc['train_mse','LinearRegression'] = mean_squared_error(y_pred = LR.predict(X_train), y_true=y_train)


# ## KNN

# In[27]:


knn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)
 
models.loc['train_mse','kNN'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)


# In[28]:


models


# ## Random Forest

# In[29]:


# buat model prediksi
RF = RandomForestRegressor(n_estimators=45, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)


# In[30]:


models


# # 5. Model Evaluation

# In[31]:


# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])


# In[32]:


# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['LR','KNN','RF'])
 
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'LR': LR, 'KNN': knn, 'RF': RF}
 
# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 
# Panggil mse
mse


# In[33]:


fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)


# In[34]:


prediksi = X_test.iloc[:2].copy()
pred_dict = {'y_true':y_test[:2]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)


# In[ ]:




