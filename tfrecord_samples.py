


##################################################################################################################################
# The TFRecord format
# The TFRecord format is a simple format for storing a sequence of binary records. 
# Protocol buffers are a cross-platform, cross-language library for efficient serialization of structured data. 
# Protocol messages are defined by .proto files, these are often the easiest way to understand a message type.

# The tf.Example message (or protobuf) is a flexible message type that represents a {"string": value} mapping. 
# It is designed for use with TensorFlow and is used throughout the higher-level APIs such as TFX. Note: While useful, these structures are optional. 
# There is no need to convert existing code to use TFRecords, unless you are using tf.data and reading data is still the bottleneck to training. 
# See Data Input Pipeline Performance for dataset performance tips.

####################################################
# tf.Example
# Data types for tf.Example
# Fundamentally, a tf.Example is a {"string": tf.train.Feature} mapping.
####################################################

# The tf.train.Feature message type can accept one of the following three types (See the .proto file for reference). 
# Most other generic types can be coerced into one of these:

# tf.train.BytesList (the following types can be coerced)

# string
# byte
# tf.train.FloatList (the following types can be coerced)

# float (float32)
# double (float64)
# tf.train.Int64List (the following types can be coerced)

# bool
# enum
# int32
# uint32
# int64
# uint64

##################################################################################################################################

# You can use any Python source file as a module by executing an import statement in some other Python source file.
# The import statement combines two operations; it searches for the named module, then it binds the results of that search
# to a name in the local scope.
#!pip install --upgrade tensorflow==2.5


import tensorflow as tf

import numpy as np
import IPython.display as display

print("TensorFlow version: ",tf.version.VERSION)


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))


feature = _float_feature(np.exp(1))

# `SerializeToString()` serializes the message and returns it as a string
feature.SerializeToString()



## create a sample dataset of 4 features here 
# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)



def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()



## sample 
example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
serialized_example

example_proto = tf.train.Example.FromString(serialized_example)
example_proto


### create a features_dataset
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
features_dataset



for f0,f1,f2,f3 in features_dataset.take(1):
  print(f0)
  print(f1)
  print(f2)
  print(f3)


def tf_serialize_example(f0,f1,f2,f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0,f1,f2,f3),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar



# `.map` function maps across the elements of the dataset.
serialized_features_dataset = features_dataset.map(tf_serialize_example)
serialized_features_dataset


def generator():
  for features in features_dataset:
       yield serialize_example(*features)


# Create a Dataset whose elements are generated by generator using `.from_generator` function
serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())


# writing tfrecord file 
##################################################### 

filename = 'test.tfrecord'
# `.TFRecordWriter` function writes a dataset to a TFRecord file
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)





#####################################################
# Reading the tfrecord file 
##################################################### 

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset


# Use the `.take` method to pull ten examples from the dataset.
for raw_record in raw_dataset.take(10):
  print(repr(raw_record))



## These tensors can be parsed using the function below. 
# Note that the feature_description is necessary here because datasets use graph-execution, and 
# need this description to build their shape and type signature:

# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)



parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset


for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))




#####################################################
### Write and Read TFRecord file ###
#####################################################

# Write the `tf.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)