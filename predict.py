from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os
from PIL import Image
# model = load_model('my_model.h5')
model = load_model('models/model_v7.h5')

# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(loss='binary_crossentropy', optimizer="adam",metrics = ['accuracy'])

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


data_test, y_test = load_and_resize_image(2, 'test', (150,150), False)
print(data_test.shape)
print(y_test)
#predict the result
y_pred = []
# for i in data_test:
# print(i.shape)
result = model.predict(data_test)
threshold = 0.5
y_pred = np.where(result > threshold, 1, 0)
print('y_pred', y_pred)
print('y_test', y_test)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)