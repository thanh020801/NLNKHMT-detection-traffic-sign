from tensorflow.keras.models import load_model
import numpy as np
model = load_model('./models/model_v7.h5')

import cv2
labels = {0: "40km/h", 1: "50km/h"}
cap = cv2.VideoCapture(cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    # Lấy khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Tiền xử lý khung hình
    # processed_frame = preprocess(frame)
    img = cv2.resize(frame, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    # Dự đoán biển báo giao thông từ khung hình đã tiền xử lý
    prediction = model.predict(img)
    threshold = 0.5
    prediction = np.where(prediction > threshold, 1, 0)
    prediction = np.array(prediction).flatten()[0]
    # prediction = 1
    # Hiển thị kết quả dự đoán trên khung hình
    cv2.putText(frame, 'Prediction: {}'.format(labels[prediction]), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình đã được dự đoán lên màn hình
    cv2.imshow('frame', frame)

    # Nhấn 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

