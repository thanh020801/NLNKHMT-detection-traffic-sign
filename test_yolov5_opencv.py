
# from tensorflow.keras.models import load_model
# from ultralytics import YOLO
# import numpy as np
# # from PIL import ImageFont, ImageDraw, Image
# import cv2

# # tìm font có hỗ trợ tiếng Việt
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# font_color = (0, 0, 255)
# font_thickness = 1


# model_detection = YOLO(r"runs\detect\train6\weights\best.pt")
# model = load_model('./models/v5.h5')

# labels = ["Gioi han toc do 40km/h", "Gioi han toc do 50km/h", "Gioi han toc do 60km/h",
#           "Het han Gioi han toc do toi đa 40km/h", "Het han Gioi han toc do toi đa 50km/h",
#           "Het han Gioi han toc do toi đa", "Vao khu vuc dan cu", "Ra khoi khu vuc dan cu", "Cam", "Stop",
#           "Cho"
# ]

# def format_frame(src):
#     img = cv2.resize(src, (150, 150))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.expand_dims(img, axis=0)
#     return img
# # path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\5\2023-02-27-112747070.mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\3\Untitled video - Made with Clipchamp (14).mp4"
# # path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\7\VID_20230305_173249.mp4"
# # path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\4\Untitled video - Made with Clipchamp (2).mp4"
# # path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\9\Untitled video - Made with Clipchamp.mp4"
# # path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\10\VID_20230222_171845.mp4"
# # path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\10\VID_20230205_165732.mp4"
# # path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\3\Untitled video - Made with Clipchamp (5).mp4"
# cap = cv2.VideoCapture(path)
# # check = ""
# while True:

#     ret, frame = cap.read()
#     if not ret:
#         break
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = model_detection.predict(source=img)
#     # xyxy (top-left x, top-left y, bottom-right x, bottom-right y)
#     if len(results[0].boxes.xyxy) > 0:
#         x1,y1,x2,y2= results[0].boxes.xyxy[1].numpy()
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(img.shape[1], x2)
#         y2 = min(img.shape[0], y2)

#         new_frame = frame[int(y1):int(y2), int(x1):int(x2)]

#         img = format_frame(new_frame)
#         pred = model.predict(img)
#         classed = np.argmax(pred)
#         print('classes', classed)
#         # print('pred[0].max()', np.argmax(pred))
#         # cv2.imwrite('test/a.jpg',new_frame)
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(frame, labels[classed], (100,100),  font, font_scale, font_color, font_thickness, cv2.LINE_AA)

#         # cv2.putText(frame, labels[classed], (int(x1), int(y1) - 10),  font, font_scale, font_color, font_thickness, cv2.LINE_AA)
#     cv2.imshow('predict', frame)
#     # if check == '':
#     #     check = input('cho video chay')
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


















from tensorflow.keras.models import load_model
from ultralytics import YOLO
import numpy as np
# from PIL import ImageFont, ImageDraw, Image
import cv2

threshold = 0.77
# tìm font có hỗ trợ tiếng Việt
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)
font_thickness = 2


model_detection = YOLO(r"runs\detect\train6\weights\best.pt")
model = load_model('./models/mega_v1.h5')

labels = ["Gioi han toc do 40km/h", "Gioi han toc do 50km/h", "Gioi han toc do 60km/h",
          "Het han toc do toi da 40km/h", "Het han toc do toi da 50km/h",
          "Het han toc do toi da", "Vao khu vuc dan cu", "Ra khoi khu vuc dan cu", "Cam", "Stop",
          "Cho"
]

def format_frame(src):
    img = cv2.resize(src, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img


# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_164033.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_163248.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_163400.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_163441.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_163516.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_171406.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_171503.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_171626.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_171652.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_172201.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_172611.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_172759.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_165037.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_170949.mp4"
# path= r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\test\VID_20230326_171022.mp4"

# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\5\2023-02-27-112747070.mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\3\Untitled video - Made with Clipchamp (14).mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\7\VID_20230305_173249.mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\4\Untitled video - Made with Clipchamp (2).mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\9\Untitled video - Made with Clipchamp.mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\10\VID_20230222_171845.mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\10\VID_20230205_165732.mp4"
# path = r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\3\Untitled video - Made with Clipchamp (5).mp4"
cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:

    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model_detection.predict(source=img, verbose=False)
    # xyxy (top-left x, top-left y, bottom-right x, bottom-right y)
    if len(results[0].boxes.xyxy) > 0:
        
        for i in range(len(results[0].boxes.xyxy)):
            # print(results[0].boxes.xyxy[i])
            # print('results[0].boxes.conf[0]',results[0].boxes.conf[0])
            if results[0].boxes.conf[0] > threshold:
                x1,y1,x2,y2= results[0].boxes.xyxy[i].numpy()
                x1 = int(max(0, x1))
                y1 = int(max(0, y1))
                x2 = int(min(img.shape[1], x2))
                y2 = int(min(img.shape[0], y2))

                new_frame = frame[y1:y2, x1:x2]
                img_resize = format_frame(new_frame)
                pred = model.predict(img_resize, verbose=0)
                print(pred[0])
                access_threshold = np.amax(pred[0],axis=0)
                # print('pred', pred)
                # print('access_threshold',access_threshold)
                classed = np.argmax(pred[0])
                # print('classes', classed)
                # print('pred[0].max()', np.argmax(pred))
                # cv2.imwrite('test/a.jpg',new_frame)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame, labels[classed], (100,100),  font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                cv2.putText(frame, str(classed) +":"+ labels[classed], (int(x1), int(y1) - 10),  font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    
    frame = cv2.resize(frame, (int(width/1.5), int(height/1.5)))
    cv2.imshow('predict', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




















# import cv2

# # đọc mô hình từ tệp ONNX
# # net = cv2.dnn.readNetFromONNX(r"runs\detect\train4\weights\best.onnx")

# image = cv2.imread(r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\sources\full_size_data\3\3_0000.jpg")

# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(500,500), mean=(0, 0, 0), swapRB=False, crop=False)

# # net.setInput(blob)
# # output = net.forward()

# # print(output)



# # import tensorflow as tf

# # # Load the ONNX model into TensorFlow Graph
# # onnx_model = r"runs\detect\train4\weights\best.onnx"
# # graph_def = tf.compat.v1.GraphDef()
# # with tf.io.gfile.GFile(onnx_model, 'rb') as f:
# #     graph_def.ParseFromString(f.read())
# # tf.import_graph_def(graph_def, name='')

# # # Get the input and output tensors
# # input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('input:0')
# # output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('output:0')

# # # Run the model on some input data
# # with tf.compat.v1.Session() as sess:
# #     output = sess.run(output_tensor, feed_dict={input_tensor: blob})
# # print(output)




# import torch

# # Load the PyTorch model from a .pt file
# model_path = r"runs\detect\train4\weights\best.pt"
# model = torch.load(model_path)['model']
# model = model.float()



# # Resize the image to match the input size of the model
# resized_img = cv2.resize(image, (512, 512))


# input_data = torch.from_numpy(resized_img.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)

# with torch.no_grad():
#     output = model(input_data)

# # print(output)
# class_ids = output[0][:, -1].int()
# conf_scores = output[0][:, -2]
# bbox = output[0][:, :-2]


# threshold = 0.5
# detections = bbox[conf_scores >= threshold]
# class_ids = class_ids[conf_scores >= threshold]
# conf_scores = conf_scores[conf_scores >= threshold]


# print('class_ids', class_ids)
# print('conf_scores', conf_scores)
# print('bbox', detections)
# # # Perform inference on some input data
# # input_data = torch.randn(16, 3, 512, 512, device=torch.device('cpu'))
# # input_data = input_data.float()
# # with torch.no_grad():
# #     output = model(input_data)

# # # Convert the output to a numpy array
# # output = output[0].cpu().numpy()
# # print(output)