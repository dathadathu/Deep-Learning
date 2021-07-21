# %%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import callbacks, layers
from tensorflow.python.keras.layers.core import Flatten

# %%
(X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

len(X_train)

len(X_test)

# %%
X_train[0].shape

X_train[0]

# %%
plt.matshow(X_train[0])

# %%
y_train[3]

# %%
X_train = X_train / 255
X_test = X_test / 255


# %%
X_train_flatten = X_train.reshape(len(X_train),28*28)
X_train_flatten.shape

# %%
X_test_flatten = X_test.reshape(len(X_test),28*28)
X_test_flatten.shape

# %%
from tensorflow import keras

# %%
model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,),activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(X_train_flatten,y_train,epochs=10)


# %%
model.evaluate(X_test_flatten,y_test)

# %%
y_predicted = model.predict(X_test_flatten)

# %%
y_predicted[0]

np.argmax(y_predicted[0])
# %%
plt.matshow(X_test[0])

# %%
y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]

# %%
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm

# %%
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot = True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")


# %%
model = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784,),activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(X_train_flatten,y_train,epochs=10)
# %%
model.evaluate(X_test_flatten,y_test)
# %%
y_predicted = model.predict(X_test_flatten)

# %%
y_predicted[0]

np.argmax(y_predicted[0])
# %%
plt.matshow(X_test[0])

# %%
y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]

# %%
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm

# %%
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot = True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")


# %%
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/",histogram_freq=1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

model.fit(X_train,y_train,epochs=10,callbacks=[tb_callback])
# %%
model.evaluate(X_test,y_test)

# %%
 