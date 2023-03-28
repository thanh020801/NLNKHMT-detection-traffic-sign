# import os
# import numpy as np
# train_path = "full_size_data"

# classes = 11

# labels = []

# for i in range(classes):
#     path = os.path.join(train_path, str(i))
#     files = os.listdir(path)
#     for filename in files:
#         try:
#             if filename.endswith(".txt"):
#                 s = ""
                
#                 path_in_file = os.path.join(path, filename)
#                 # print(path_in_file)
#                 content = np.loadtxt(path_in_file)
#                 # # content[0] = 0
#                 print(content)
#                 if len(content) == 2:
#                     content[0][0] = 0
#                     content[1][0] = 0
#                     s = str(int(content[0][0])) + ' ' + str(content[0][1]) + ' '+ str(content[0][2]) + ' '+ str(content[0][3]) + ' '+ str(content[0][4]) +'\n' + str(int(content[1][0])) + ' '+ str(content[1][1]) + ' '+ str(content[1][2]) + ' '+ str(content[1][3]) + ' '+ str(content[1][4])
#                     # print(s)
#                 else:
#                     content[0] = 0
#                     s = str(int(content[0])) + ' ' + str(content[1]) + ' '+ str(content[2]) + ' '+ str(content[3]) + ' '+ str(content[4])
#                     # print(s)
#                 # with open(path_in_file, 'w') as f:
#                 #     f.write(s)
#         except:
#             print('khong the load file:', path_in_file)
#             # exit()







import os

# path = os.path.join(train_path, str(i))


path_img_train = r'E:\full_size_data_1_class\images\train'
path_img_val = r'E:\full_size_data_1_class\images\val'
path_label_train = r'E:\full_size_data_1_class\labels\train'
path_label_val = r'E:\full_size_data_1_class\labels\val'

# p1 = os.path.join(path_img_train, str(i))
# p2 = os.path.join(path_img_val, str(i))
# p3 = os.path.join(path_label_train, str(i))
# p4 = os.path.join(path_label_val, str(i))



l1 = os.listdir(path_img_train)
l2 = os.listdir(path_img_val)
l3 = os.listdir(path_label_train)
l4 = os.listdir(path_label_val)

print(len(l1))
print(len(l2))
print(len(l3))
print(len(l4))