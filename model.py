import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

df = pd.read_csv('heart.csv')

# Converting categorical features to one-hot
chest_pain_type = df.pop('cp')
df['typical angina'] = (chest_pain_type == 0) * 1.0
df['atypical angina'] = (chest_pain_type == 1) * 1.0
df['non-anginal pain'] = (chest_pain_type == 2) * 1.0
df['asymptomatic'] = (chest_pain_type == 3) * 1.0
thal = df.pop('thal')
df['thal normal'] = (thal == 1) * 1.0
df['thal fixed defect'] = (thal == 2) * 1.0
df['thal reversable defect'] = (thal == 3) * 1.0

# Splitting data into training and validation sets
df_train = df.sample(frac=0.8)
df_val = df.drop(df_train.index)

# Extracting labels
y_train = df_train.pop('target').values.astype(np.int32)
y_val = df_val.pop('target').values.astype(np.int32)

# Normalizing features
train_stats = df_train.describe()
train_stats = train_stats.transpose()


def normalize(x):
	return (x - train_stats['mean']) / train_stats['std']


df_train = normalize(df_train)
df_val = normalize(df_val)

# Extracting feature data
x_train = df_train.values.astype(np.float64)
x_val = df_val.values.astype(np.float64)

# Defining a neural network, compiling, and fitting
model = keras.models.Sequential([
	keras.layers.Dense(15, activation='relu', input_dim=x_train.shape[1]),
	keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy']
)
history = model.fit(
	x_train,
	y_train,
	epochs=150,
	validation_data=(x_val, y_val),
	verbose=0
)

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)


# Plotting loss
# -- "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# -- b is for "solid blue line"
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('loss.png', dpi=500)
plt.show()


# Plotting accuracy
plt.clf()

# -- "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# -- b is for "solid blue line"
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('acc.png', dpi=500)
plt.show()
