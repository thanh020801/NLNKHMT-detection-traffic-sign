from tensorflow.keras.models import load_model
import numpy as np
model = load_model('./models/model_v8.h5')

import cv2
labels = ["Giới hạn tốc độ 40km/h", "Giới hạn tốc độ 50km/h", "Giới hạn tốc độ 60km/h",
          "Hết hạn giới hạn tốc độ tối đa 40km/h", "Hết hạn giới hạn tốc độ tối đa 50km/h",
          "Hết hạn giới hạn tốc độ tối đa", "Vào khu vực khu dân cư", "Ra khỏi khu vực khu dân cư", "Cấm", "Stop",
          "Chợ"
]
# labels = {0: "40km/h", 1: "50km/h"}
path =  r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video bo gh 40\VIDEO_DOWNLOAD_1675224868116_1675224945599.mp4"
#path =  r "C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video bo gh 40\VID_20230205_173149 (2).mp4"
#path =  r "C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video bo gh 40\VID_20230205_172740 (2).mp4"
#path =  r "C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video bo gh 40\VIDEO_DOWNLOAD_1675224953959.mp4"
#path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video bo gh 40\VID_20230112_102931 (2).mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video bo gh 40\VID_20230205_174617 (2).mp4"
cap = cv2.VideoCapture(path)

step = 100
while True:
    # Lấy khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (500, 500))
    # height, width = frame.shape[:2]
    # step_height = int(height / step +1 )
    # step_width = int(width / step +1)
    # # print(step_height, step_width)
    # result = ""
    # for i in range(step_height +1):
    #     for j in range(step_width +1):
    #         new_frame = frame[i:(i+1)* step, j:(j+1)*step]
    #         img = cv2.resize(new_frame, (150, 150))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = np.expand_dims(img, axis=0)
    #         # print('new_frame.shape', new_frame)
    #         prediction = model.predict(img, verbose=0)
    #         prediction = np.argmax(prediction)
    #         result = labels[prediction]
    #         break
    # print('labels: ', result)


    # cv2.imshow('frame', frame)








    # Tiền xử lý khung hình
    # processed_frame = preprocess(frame)
    img = cv2.resize(frame, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    # Dự đoán biển báo giao thông từ khung hình đã tiền xử lý
    prediction = model.predict(img, verbose=0)
    threshold = 0.7
    prediction = np.where(prediction > threshold, 1, 0)
    if prediction[0].max() != 0:
        print(prediction)
        prediction = np.argmax(prediction)
        print('label: ', labels[prediction])
    # else:
        
    # 
    # prediction = np.where(prediction > threshold, 1, 0)
    # print('prediction', prediction)
    # prediction = np.array(prediction).flatten()[0]
    # prediction = 1
    # Hiển thị kết quả dự đoán trên khung hình

    # Hiển thị khung hình đã được dự đoán lên màn hình
    cv2.imshow('frame', frame)

    # Nhấn 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

