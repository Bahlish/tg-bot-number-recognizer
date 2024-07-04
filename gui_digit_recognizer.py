import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  # pip install tensorflow-datasets
import tensorflow as tf
import logging
import numpy as np
import cv2  # pip install opencv-python


tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)



  ///подключили все библиотеки
  
  
  
  
  
  
def mnist_cnn_model():
   image_size = 28
   num_channels = 1  # 1 for grayscale images
   num_classes = 10  # Number of outputs
   model = Sequential()
   model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
            padding='same', 
input_shape=(image_size, image_size, num_channels)))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
            padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
            padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Flatten())
   # Densely connected layers
   model.add(Dense(128, activation='relu'))
   # Output layer
   model.add(Dense(num_classes, activation='softmax'))
   model.compile(optimizer=Adam(), loss='categorical_crossentropy',
            metrics=['accuracy'])
   return model
   
   
   
   ///сделали сверточную сеть
   
   
   
def mnist_cnn_train(model):
   (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

   # Get image size
   image_size = 28
   num_channels = 1  # 1 for grayscale images

   # re-shape and re-scale the images data
   train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
   train_data = train_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

   # re-shape and re-scale the images validation data
   val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
   val_data = val_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

   print("Training the network...")
   t_start = time.time()

   # Start training the network
   model.fit(train_data, train_labels_cat, epochs=8, batch_size=64,
        validation_data=(val_data, val_labels_cat))

   print("Done, dT:", time.time() - t_start)

   return model
   
   
   
   /// обучили сеть
   
   
   
model = mnist_cnn_model()
mnist_cnn_train(model)
model.save('cnn_digits_28x28.h5')
  
  
  
   ///создаем и обучаем модель
   
   
   
   def cnn_digits_predict(model, image_file):
   img = cv2.imread(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ed = cv2.Canny(blur, 10, 250)

    contours, hierarchy = cv2.findContours(ed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_x, min_y, max_x, max_y = [], [], [], []
    for item in contours:
        min_x.append(np.min([x[0][0] for x in item]))
        min_y.append(np.min([x[0][1] for x in item]))
        max_x.append(np.max([x[0][0] for x in item]))
        max_y.append(np.max([x[0][1] for x in item]))

    k = 0.04
    cropped = blur[int(np.mean(min_y)) - int(img.shape[0] * k): int(np.mean(max_y)) + int(img.shape[0] * k),
                   int(np.mean(min_x)) - int(img.shape[1] * k): int(np.mean(max_x)) + int(img.shape[1] * k)]

    thresh, image_black = cv2.threshold(cropped, 140, 255, cv2.THRESH_BINARY)
    res = cv2.resize(255 - image_black, (28, 28), interpolation=cv2.INTER_AREA)

    new_image = np.array(res)
    new_image = new_image.reshape(-1, 28, 28, 1)
    new_image = new_image / 255.0
    res = model.predict([new_image])[0]

   result = model.predict_classes([img_arr])
   return result[0]

model = tf.keras.models.load_model('cnn_digits_28x28.h5')

/////////запускаем сеть

print(cnn_digits_predict(model, 'digit_4.png'))
print(cnn_digits_predict(model, 'digit_2.png'))
print(cnn_digits_predict(model, 'digit_1.png'))
print(cnn_digits_predict(model, 'digit_6.png'))
print(cnn_digits_predict(model, 'digit_9.png'))

////////проверяем результат



