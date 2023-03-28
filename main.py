import os
import numpy as np
import pandas as pd
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from sklearn.metrics import confusion_matrix


train_path = "resize_data"
# 0:20 1:30 2:50 3:60 4:70
# {0:"Giới hạn tốc độ 40km/h", 1:"Giới hạn tốc độ 50km/h"}

def load_and_resize_image(classes, data_path, target_size, grayscale):
    data = []
    labels = []
    for i in range(classes):
        path = os.path.join(data_path,str(i))
        images = os.listdir(path)
        # count = 0
        for a in images:
            try:
                # if count >=50:
                #     break
                img = load_img( 
                            os.path.join(data_path,str(i),a),
                            grayscale=grayscale,
                            color_mode="rgb",
                            target_size=target_size,
                )
                img = np.array(img)
                data.append(img)
                labels.append(i)
                # count +=1
            except:
                print('khong the load file')
    return np.array(data), np.array(labels)

classes = 2
X, y = load_and_resize_image(classes=classes, data_path='data', target_size=(150,150),grayscale=False)
print(X.shape, y.shape)


X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=100)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#one-hot encoding the labels
# y_train = to_categorical(y_train, classes)
# y_test = to_categorical(y_test, classes)
# print(y_train.shape, y_test.shape)

# def built_model(input_shape, classes):
#     model = Sequential()
#     model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(64,(3,3),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer="adam",metrics = ['accuracy'])
#     print(model.summary())
#     return model


def built_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

model = built_model(X_train.shape[1:], classes )

model.fit(x = X_train, y = y_train, batch_size=32, epochs=10)

score =  model.evaluate( x = X_test, y = y_test, batch_size=32)







   
print('test score: ', score[0])
print('test accuracy: ', score[1])



def load_and_resize_image1(classes, data_path, target_size, grayscale):

    data = []
    labels = []
    path = os.path.join(data_path)
    images = os.listdir(path)
    for a in images:
        try:
            img = load_img( 
                        os.path.join(data_path,a),
                        grayscale=grayscale,
                        color_mode="rgb",
                        target_size=target_size,
            )
            img = np.array(img)
            if len(a) == 11:
                labels.append(1)
            else:
                labels.append(0)
            data.append(img)
        except:
            print('khong the load file')
    return np.array(data) , np.array(labels)

data_test, y_test = load_and_resize_image1(2, 'test', (150,150), False)
print(data_test.shape)
print(y_test)
#predict the result
score = model.evaluate(data_test, y_test, batch_size=32)
print("score: ", score[0], score[1])


result = model.predict(data_test)
threshold = 0.5
y_pred = np.where(result > threshold, 1, 0)
print(y_pred)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)


model.save('models/model_v6.h5')
