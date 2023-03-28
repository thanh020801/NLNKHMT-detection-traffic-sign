from ultralytics import YOLO
model = YOLO()
# Load a model
# model = YOLO('bienbao.yaml')  # build a new model from scratch
# model = YOLO(r'runs\detect\train4\weights\best.pt')  # load a pretrained model (recommended for training)
# print(YOLO)
# Use the model
results = model.train(data='bienbao.yaml', epochs=20, batch = 16, imgsz = 500)  # train the model
results = model.val()  # evaluate model performance on the validation set

# results = model(r"C:\Users\ADMIN\Desktop\HK2_2023\NienLuan\video\3\Untitled video - Made with Clipchamp (5).mp4")  # predict on an image
success = model.export(format='onnx')  # export the model to ONNX format