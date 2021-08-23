# You can use any Python source file as a module by executing an import statement in some other Python source file.
# The import statement combines two operations; it searches for the named module, then it binds the results of that search
# to a name in the local scope.
import numpy as np
import pandas as pd
# Import matplotlib to visualize the model
import matplotlib.pyplot as plt
# Seaborn is a Python data visualization library based on matplotlib
import seaborn as sns
# %matplotlib inline sets the backend of matplotlib to the `inline` backend
%matplotlib inline

import tensorflow as tf


from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("TensorFlow version: ",tf.version.VERSION)



## import the csv data 
URL = 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
# Read a comma-separated values (csv) file into a DataFrame using the read_csv() function
dataframe = pd.read_csv(URL)
# Get the first five rows using the head() method
dataframe.head()


# Get a concise summary of a DataFrame
dataframe.info()



# Create test, validation and train samples from one dataframe with pandas.
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# NOTE :  we will wrap the dataframes with tf.data. 
# This will enable us to use feature columns as a bridge to map from the columns in the Pandas dataframe to features used to train a model. 
# If we were working with a very large CSV file (so large that it does not fit into memory), we would use tf.data to read it from disk directly. That is not covered in this lab.

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) # TODO 2a
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


# A small batch size is used for demonstration purposes
batch_size = 5


# TODO 2b
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)



# If you don't use take(1), all elements will eventually be fetched
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch)


## Lab Task 3: Demonstrate several types of feature column

# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]
print(example_batch)


# A utility method to create a feature column and to transform a batch of data
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())



# Create a numeric feature column out of `age`
age = feature_column.numeric_column("age")
tf.feature_column.numeric_column
print(age)


## =======================
## Bucketized columns
## =======================
# Often, you don't want to feed a number directly into the model, but instead split its value into different categories based on numerical ranges. 
# Consider raw data that represents a person's age.
# Instead of representing age as a numeric column, we could split the age into several buckets using a bucketized column.
# Notice the one-hot values below describe which age range each row matches.

# Create a bucketized feature column out of `age` with the following boundaries and demo it.
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets) # TODO 3a




## =======================
##  Categorical columns
## =======================
# Create a categorical vocabulary column out of the
# above mentioned categories with the key specified as `thal`.
thal = tf.feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

# Create an indicator column out of the created categorical column.
thal_one_hot = tf.feature_column.indicator_column(thal)
demo(thal_one_hot)



## =======================
# Embedding columns
## =======================

# Notice the input to the embedding column is the categorical column
# we previously created
# Set the size of the embedding to 8, by using the dimension parameter
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)



## =======================
# Hashed feature columns
## =======================

# Create a hashed feature column with `thal` as the key and 1000 hash buckets.
thal_hashed = tf.feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(thal_hashed))


## =======================
# Crossed feature columns
## =======================
# Create a crossed column using the bucketized column (age_buckets)
# the categorical vocabulary column (thal), and 1000 hash buckets.
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(tf.feature_column.indicator_column(crossed_feature))


## ========================================================
### Feature engineering for input to the Model 
feature_columns = []
## ========================================================

# numeric cols
# Create a feature column out of the header using a numeric column.
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
# Create a bucketized feature column out of the age column using the following boundaries.
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
# Create a categorical vocabulary column out of the below categories with the key specified as `thal`.
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
# Create an embedding column out of the categorical vocabulary
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
# Create a crossed column using the bucketized column (age_buckets),
# the categorical vocabulary column (thal), and 1000 hash buckets.
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)



## ========================================================
### Input Feature Columns to a Keras Model
## ========================================================

# Create a Keras DenseFeatures layer and pass the feature_columns
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


## Earlier, we used a small batch size to demonstrate how feature columns worked. We create a new input pipeline with a larger batch size.
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


## Create, compile, and train the model
# `Sequential` provides training and inference features on this model.
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

# `Compile` configures the model for training.
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# `Fit` trains the model for a fixed number of epochs
history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=5)


# `Evaluate` returns the loss value & metrics values for the model in test mode.
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)




## Visualize the model loss curve

def plot_curves(history, metrics):
    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(10, 5))

    for idx, key in enumerate(metrics):  
        ax = fig.add_subplot(nrows, ncols, idx+1)
        plt.plot(history.history[key])
        plt.plot(history.history['val_{}'.format(key)])
        plt.title('model {}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left');    
    
    

plot_curves(history, ['loss', 'accuracy'])