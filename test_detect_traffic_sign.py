# from tensorflow.keras.models import load_model
import numpy as np
import os
# model = load_model('./models/model_v8.h5')

import cv2

link = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\8"
path = os.path.join(link)
videos = os.listdir(path)
# print(videos)
step = 0
i = 0
for filename in videos:
    print("file name: ",os.path.join(link, filename))
    cap = cv2.VideoCapture(os.path.join(link, filename))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i%2 == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resize_frame = cv2.resize(img,(500,500))
            cv2.imwrite('./full_size_data/8/8_000'+str(step)+'.jpg', resize_frame)
            cv2.imshow('frame', resize_frame)
            step+=1 
        i+=1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # break
print(i, step)
# cap = cv2.VideoCapture(path)

# step = 0
# while True:
#     # Lấy khung hình từ camera
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cv2.imwrite('./full_size_data/3/img_'+str(step)+".jpg", frame)
#     cv2.imshow('frame', frame)
#     step+=1
#     # Nhấn 'q' để thoát khỏi vòng lặp
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Giải phóng tài nguyên
# cap.release()
# cv2.destroyAllWindows()

